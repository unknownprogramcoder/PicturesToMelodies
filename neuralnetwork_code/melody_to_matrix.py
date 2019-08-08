#!/usr/bin/env python
# -*- coding: utf8 -*-
import mido
from mido import MidiFile
from unidecode import unidecode
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl

def get_pianoroll_time(pianoroll):
    T_pr_list = []
    for k, v in pianoroll.items():
        T_pr_list.append(v.shape[0])
    if not len(set(T_pr_list)) == 1:
        print("Inconsistent dimensions in the new PR")
        return None
    return T_pr_list[0]

def get_pitch_dim(pianoroll):
    N_pr_list = []
    for k, v in pianoroll.items():
        N_pr_list.append(v.shape[1])
    if not len(set(N_pr_list)) == 1:
        print("Inconsistent dimensions in the new PR")
        raise NameError("Pr dimension")
    return N_pr_list[0]

def dict_to_matrix(pianoroll):
    T_pr = get_pianoroll_time(pianoroll)
    N_pr = get_pitch_dim(pianoroll)
    rp = np.zeros((T_pr, N_pr), dtype=np.int16)
    for k, v in pianoroll.items():
        rp = np.maximum(rp, v)
    return rp


def write_midi(pr, ticks_per_beat, write_path, tempo=80):
    def pr_to_list(pr):
        # List event = (pitch, velocity, time)
        T, N = pr.shape
        t_last = 0
        pr_tm1 = np.zeros(N)
        list_event = []
        for t in range(T):
            pr_t = pr[t]
            mask = (pr_t != pr_tm1)
            if (mask).any():
                for n in range(N):
                    if mask[n]:
                        pitch = n
                        velocity = int(pr_t[n])
                        # Time is incremented since last event
                        t_event = t - t_last
                        t_last = t
                        list_event.append((pitch, velocity, t_event))
            pr_tm1 = pr_t
        return list_event
    # Tempo
    microseconds_per_beat = mido.bpm2tempo(tempo)
    # Write a pianoroll in a midi file
    mid = MidiFile()
    mid.ticks_per_beat = ticks_per_beat

    # Each instrument is a track
    for instrument_name, matrix in pr.items():
        # Add a new track with the instrument name to the midi file
        track = mid.add_track(instrument_name)
        # transform the matrix in a list of (pitch, velocity, time)
        events = pr_to_list(matrix)
        # Tempo
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        # Add the program_change
        try:
            program = program_change_mapping[instrument_name]
        except:
            # Defaul is piano
            # print instrument_name + " not in the program_change mapping"
            # print "Default value is 1 (piano)"
            # print "Check acidano/data_processing/utils/program_change_mapping.py"
            program = 1
        track.append(mido.Message('program_change', program=program))

        # This list is required to shut down
        # notes that are on, intensity modified, then off only 1 time
        # Example :
        # (60,20,0)
        # (60,40,10)
        # (60,0,15)
        notes_on_list = []
        # Write events in the midi file
        for event in events:
            pitch, velocity, time = event
            if velocity == 0:
                # Get the channel
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
                notes_on_list.remove(pitch)
            else:
                if pitch in notes_on_list:
                    track.append(mido.Message('note_off', note=pitch, velocity=0, time=time))
                    notes_on_list.remove(pitch)
                    time = 0
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
                notes_on_list.append(pitch)
    mid.save(write_path)
    return

#######
# Pianorolls dims are  :   TIME  *  PITCH


class Read_midi(object):
    def __init__(self, song_path, quantization):
        ## Metadata
        self.__song_path = song_path
        self.__quantization = quantization

        ## Pianoroll
        self.__T_pr = None

        ## Private misc
        self.__num_ticks = None
        self.__T_file = None

    @property
    def quantization(self):
        return self.__quantization

    @property
    def T_pr(self):
        return self.__T_pr

    @property
    def T_file(self):
        return self.__T_file

    def get_total_num_tick(self):
        # Midi length should be written in a meta message at the beginning of the file,
        # but in many cases, lazy motherfuckers didn't write it...

        # Read a midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)

        # Parse track by track
        num_ticks = 0
        for i, track in enumerate(mid.tracks):
            tick_counter = 0
            for message in track:
                # Note on
                time = float(message.time)
                tick_counter += time
            num_ticks = max(num_ticks, tick_counter)
        self.__num_ticks = num_ticks

    def get_pitch_range(self):
        mid = MidiFile(self.__song_path)
        min_pitch = 200
        max_pitch = 0
        for i, track in enumerate(mid.tracks):
            for message in track:
                if message.type in ['note_on', 'note_off']:
                    pitch = message.note
                    if pitch > max_pitch:
                        max_pitch = pitch
                    if pitch < min_pitch:
                        min_pitch = pitch
        return min_pitch, max_pitch

    def get_time_file(self):
        # Get the time dimension for a pianoroll given a certain quantization
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat
        # Total number of ticks
        self.get_total_num_tick()
        # Dimensions of the pianoroll for each track
        self.__T_file = int((self.__num_ticks / ticks_per_beat) * self.__quantization)
        return self.__T_file

    def read_file(self):
        # Read the midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat

        # Get total time
        self.get_time_file()
        T_pr = self.__T_file
        # Pitch dimension
        N_pr = 128
        pianoroll = {}

        def add_note_to_pr(note_off, notes_on, pr):
            pitch_off, _, time_off = note_off
            # Note off : search for the note in the list of note on,
            # get the start and end time
            # write it in th pr
            match_list = [(ind, item) for (ind, item) in enumerate(notes_on) if item[0] == pitch_off]
            if len(match_list) == 0:
                print("Try to note off a note that has never been turned on")
                # Do nothing
                return

            # Add note to the pr
            pitch, velocity, time_on = match_list[0][1]
            pr[time_on:time_off, pitch] = velocity
            # Remove the note from notes_on
            ind_match = match_list[0][0]
            del notes_on[ind_match]
            return

        # Parse track by track
        counter_unnamed_track = 0
        for i, track in enumerate(mid.tracks):
            # Instanciate the pianoroll
            pr = np.zeros([T_pr, N_pr])
            time_counter = 0
            notes_on = []
            for message in track:

                ##########################################
                ##########################################
                ##########################################
                # TODO : keep track of tempo information
                # import re
                # if re.search("tempo", message.type):
                #     import pdb; pdb.set_trace()
                ##########################################
                ##########################################
                ##########################################


                # print message
                # Time. Must be incremented, whether it is a note on/off or not
                time = float(message.time)
                time_counter += time / ticks_per_beat * self.__quantization
                # Time in pr (mapping)
                time_pr = int(round(time_counter))
                # Note on
                if message.type == 'note_on':
                    # Get pitch
                    pitch = message.note
                    # Get velocity
                    velocity = message.velocity
                    if velocity > 0:
                        notes_on.append((pitch, velocity, time_pr))
                    elif velocity == 0:
                        add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)
                # Note off
                elif message.type == 'note_off':
                    pitch = message.note
                    velocity = message.velocity
                    add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)

            # We deal with discrete values ranged between 0 and 127
            #     -> convert to int
            pr = pr.astype(np.int16)
            if np.sum(np.sum(pr)) > 0:
                name = unidecode(track.name)
                name = name.rstrip('\x00')
                if name == u'':
                    name = 'unnamed' + str(counter_unnamed_track)
                    counter_unnamed_track += 1
                if name in pianoroll.keys():
                    # Take max of the to pianorolls
                    pianoroll[name] = np.maximum(pr, pianoroll[name])
                else:
                    pianoroll[name] = pr
        return pianoroll

if __name__ == '__main__':
    the_number_of_file1 = 10
    the_number_of_file2 = 2
    time_len = 256
    
    arrays1 = np.zeros((the_number_of_file1, time_len)) 
    aaa = None
    for i in range(the_number_of_file1):
        filepath = "C:\\Users\\kebinxyz\\Desktop\\project_pts\\melodies_midi\\m{}.mid".format(i+1)
        aaa = Read_midi(filepath, 8).read_file()
        bbb = dict_to_matrix(aaa)
        ccc = bbb[0:time_len, 40:90] #자르기.
        #원핫벡터->인덱스로 바꾸기. (그냥 단선율로만)
        ddd = np.argmax(ccc, axis=1)
        plt.figure(i+1, figsize = (4, 8))
        pp = np.transpose(ccc*(0.01))
        ppp = np.flip(pp, 0)
        plt.imshow(ppp)
        arrays1[i] = np.expand_dims(ddd, axis=0)
    print(arrays1.shape)
    with open('melodies_for_decoder_input_train', 'wb') as f:
        pkl.dump(arrays1, f)
    
    plt.figure(the_number_of_file1 + 1, figsize = (4, 8))
    plt.imshow(np.zeros((1,1)))
    
    arrays2 = np.zeros((the_number_of_file2, time_len)) 
    aaa = None
    for i in range(the_number_of_file2):
        filepath = "C:\\Users\\kebinxyz\\Desktop\\project_pts\\melodies_midi\\t{}.mid".format(i+1)
        aaa = Read_midi(filepath, 8).read_file()
        bbb = dict_to_matrix(aaa)
        ccc = bbb[0:time_len, 40:90] #자르기.
        #원핫벡터->인덱스로 바꾸기. (그냥 단선율로만)
        ddd = np.argmax(ccc, axis=1)
        plt.figure(the_number_of_file1 + 1 + i + 1, figsize = (4, 8))
        pp = np.transpose(ccc*(0.01))
        ppp = np.flip(pp, 0)
        plt.imshow(ppp)
        arrays2[i] = np.expand_dims(ddd, axis=0)
    print(arrays2.shape)
    with open('melodies_for_decoder_input_test', 'wb') as f:
        pkl.dump(arrays2, f)