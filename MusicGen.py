
# coding: utf-8

# separate left hand and right hand parts, choice of chords or single/double notes

# In[24]:

import numpy as np
from MidiFile3 import MIDIFile
from scipy import stats
from os import walk, makedirs
from os.path import isdir
from pythonosc import osc_message_builder
from pythonosc import udp_client
import argparse

# In[25]:

notes = ['A', 'Bb', 'B', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'silence']
white_notes = [0, 2, 3, 5, 7, 8, 10]
black_notes = [1, 4, 6, 9, 11]
composition = []
lengths = {'1/4': 1, '1/8': 0.5, '1/16': 0.25, '3/8': 1.5, '1/2': 2, '1': 4, '2': 8, '4': 16}
lengths_list = [1, 0.5, 0.25, 1.5, 2, 4, 8, 16]
volumes_list = [60, 70, 80, 90, 100, 110]
shift_map = {'A': -3,
             'Bb': -2,
             'B': -1,
             'C': 0,
             'C#': 1,
             'D': 2,
             'Eb': 3,
             'E': 4,
             'F': 5,
             'F#': 6,
             'G': 7,
             'G#': 8}
sequins_signes = ['u', 'd', 'r', 'R', 't2', 't3', 't4',
                  'ou', 'ou2', 'od', 'od2', ]  # up, down, repeat note, repeat even if silence intended,
                                               # trills, octave skip from above


# In[76]:

Bbmin = {'A': 0,
                         'Bb': 1.,
                         'B': 0,
                         'C': 1,
                         'C#': 1.,
                         'D': 0,
                         'Eb': 1.,
                         'E': 0,
                         'F': 1,
                         'F#': 1.,
                         'G': 0,
                         'G#': 1.,
                         }


CMaj = {'A': 1,
                         'Bb': 0.,
                         'B': 1,
                         'C': 1,
                         'C#': 0.,
                         'D': 1,
                         'Eb': 0.,
                         'E': 1,
                         'F': 1,
                         'F#': 0.,
                         'G': 1,
                         'G#': 0.,
                         }

Dmin = {'A': 1,
                         'Bb': 1.,
                         'B': 0,
                         'C': 1,
                         'C#': 0.,
                         'D': 1,
                         'Eb': 0.,
                         'E': 1,
                         'F': 1,
                         'F#': 0.,
                         'G': 1,
                         'G#': 0.,
                         }

MinorChord = {'A': 0,
                         'Bb': 0.,
                         'B': 1,
                         'C': 0,
                         'C#': 0.,
                         'D': 0,
                         'Eb': 0.,
                         'E': 1,
                         'F': 0,
                         'F#': 0.,
                         'G': 1,
                         'G#': 0.,
                         }
MajorChord = {'A': 0,
                         'Bb': 0.,
                         'B': 1,
                         'C': 0,
                         'C#': 0.,
                         'D': 0,
                         'Eb': 0.,
                         'E': 1,
                         'F': 0,
                         'F#': 0.,
                         'G': 0,
                         'G#': 1.,
                         }

Chromatic = {'A': 1,
                         'Bb': 1.,
                         'B': 1,
                         'C': 1,
                         'C#': 1.,
                         'D': 1,
                         'Eb': 1.,
                         'E': 1,
                         'F': 1,
                         'F#': 1.,
                         'G': 1,
                         'G#': 1.,
                         }



FSharp = {'A': 0,
                         'Bb': 0.,
                         'B': 0.,
                         'C': 0,
                         'C#': 0.,
                         'D': 0,
                         'Eb': 0.,
                         'E': 0.,
                         'F': 0,
                         'F#': 1.,
                         'G': 0,
                         'G#': 0.,
                         }

FSharpHMinor = {'A': 1,
                         'Bb': 0.,
                         'B': 1.,
                         'C': 0,
                         'C#': 1.,
                         'D': 1,
                         'Eb': 0.,
                         'E': 0.,
                         'F': 1,
                         'F#': 1.,
                         'G': 0,
                         'G#': 1.,
                         }


FSharpHMinor_noroot = {'A': 1,
                         'Bb': 0.,
                         'B': 1.,
                         'C': 0,
                         'C#': 1.,
                         'D': 1,
                         'Eb': 0.,
                         'E': 0.,
                         'F': 1,
                         'F#': 0.,
                         'G': 0,
                         'G#': 1.,
                         }

Gmin = {'A': 1,
                         'Bb': 1.,
                         'B': 0,
                         'C': 1,
                         'C#': 0.,
                         'D': 1,
                         'Eb': 1.,
                         'E': 0,
                         'F': 1,
                         'F#': 0.,
                         'G': 1,
                         'G#': 0.,
                         }

GSharpHMinor = {'A': 0,
                         'Bb': 1.,
                         'B': 1,
                         'C': 0,
                         'C#': 1.,
                         'D': 0,
                         'Eb': 1.,
                         'E': 1,
                         'F': 0,
                         'F#': 0.,
                         'G': 1,
                         'G#': 1.,
                         }

Fmin = {'A': 0,
                         'Bb': 1.,
                         'B': 0,
                         'C': 1,
                         'C#': 1.,
                         'D': 0,
                         'Eb': 1.,
                         'E': 0,
                         'F': 1,
                         'F#': 0.,
                         'G': 1,
                         'G#': 1.,
                         }


# In[77]:

class Note:
    
    def __init__(self, value, octave, start_time, duration, volume):
        self.value = value
        self.octave = octave
        self.start_time = start_time
        self.duration = duration
        self.volume = volume


# In[78]:

def filt(x):
    if x <= 0.:
        return 0.
    else:
        return x


# In[79]:

def add_silence(distr, sparsness):
    res = {note: distr[note] * (1. - sparsness) for note in notes[:12]}
    res['silence'] = sparsness
    return res


# In[80]:

def convert_pitch_to_MIDI(value, octave):
    return (octave + 2) * 12 + shift_map[value]


# In[81]:

def check_dissonance(note1, note2):
    """
    returns true if dissonant (semitone or tritone apart)
    """
    v1 = shift_map[note1]
    v2 = shift_map[note2]
    if abs(v1 - v2) == 1 or abs(v1 - v2) == 11:  # semitone apart
        return True
    if abs(v1 - v2) == 6:
        return True
    return False


# In[82]:

def get_note_distr(start_probs, end_probs, sparsness_start, sparsness_end, time, composition_length):    
    new_distr = {}
    
    unnormed_prob = 0.
    
    tc = time / composition_length
    
    for note in notes[:12]:
        new_distr[note] = filt(start_probs[note] + (end_probs[note] - start_probs[note]) * tc)
        unnormed_prob += filt(start_probs[note] + (end_probs[note] - start_probs[note]) * tc)
        
    for note in notes[:12]:
        new_distr[note] /= unnormed_prob
    
    full_distr = add_silence(new_distr, sparsness_start + (sparsness_end - sparsness_start) * tc)
    probs = [full_distr[note] for note in notes]
    distrib = stats.rv_discrete(name='noted', values=(np.arange(13), probs))
    return distrib


# In[83]:

def get_note_length_distr(start_probs, end_probs, time, composition_length):
    length_probs = {}
    
    unnormed_prob = 0.
    
    tc = time / composition_length
    
    for length in lengths:
        length_probs[length] = start_probs[length] + (end_probs[length] - start_probs[length]) * tc
        unnormed_prob += start_probs[length] + (end_probs[length] - start_probs[length]) * tc
    
    tmp_l_probs = []
    vals_list = [[k, v] for k, v in length_probs.items()]
    for l in lengths_list:
        for le, pr in vals_list:
            if lengths[le] == l:
                tmp_l_probs.append(pr / unnormed_prob)
    distrib = stats.rv_discrete(name='lengthd', values=(np.arange(8), tmp_l_probs))
    return distrib


# In[84]:

def get_octave_distr(start_probs, end_probs, time, composition_length):
    octave_probs = []
    
    unnormed_prob = 0.
    
    tc = time / composition_length
    
    for o in range(7):
        octave_probs.append([o + 1, start_probs[o][1] + (end_probs[o][1] - start_probs[o][1]) * tc])
        unnormed_prob += start_probs[o][1] + (end_probs[o][1] - start_probs[o][1]) * tc
    
    distrib = stats.rv_discrete(name='octd', values=([o[0] for o in octave_probs], [o[1] / unnormed_prob
                                                                                    for o in octave_probs]))
    return distrib


# In[85]:

def get_piece_length_distr(start_probs, end_probs, time, composition_length):
    length_probs = {}
    
    unnormed_prob = 0.
    
    tc = time / composition_length
    
    for length in lengths:
        length_probs[length] = start_probs[length] + (end_probs[length] - start_probs[length]) * tc
        unnormed_prob += start_probs[length] + (end_probs[length] - start_probs[length]) * tc
    
    tmp_l_probs = []
    vals_list = [[k, v] for k, v in length_probs.items()]
    for l in lengths_list:
        for le, pr in vals_list:
            if lengths[le] == l:
                tmp_l_probs.append(pr / unnormed_prob)
    distrib = stats.rv_discrete(name='pieceld', values=(np.arange(8), tmp_l_probs))
    return distrib


# In[86]:

def get_chord_single_distr(start_prob, end_prob, time, composition_length):
    cs_probs = []
    
    unnormed_prob = 0.
    
    tc = time / composition_length
    
    chord_prob = filt(start_prob + (end_prob - start_prob) * tc)
    single_prob = 1. - chord_prob
    
    distrib = stats.rv_discrete(name='csd', values=([0, 1], [chord_prob, single_prob]))
    return distrib


# In[87]:

def get_vel_var(distr='uniform', std=5):
    return -std + stats.randint.rvs(0, 2*std+1)


# In[88]:

def get_vel_distr(start_probs, end_probs, time, composition_length):
    vel_probs = []
    
    unnormed_prob = 0.
    
    tc = time / composition_length
    
    for o in range(len(start_probs)):
        vel_probs.append([o, start_probs[o][1] + (end_probs[o][1] - start_probs[o][1]) * tc])
        unnormed_prob += start_probs[o][1] + (end_probs[o][1] - start_probs[o][1]) * tc
    
#     print(vel_probs)
    distrib = stats.rv_discrete(name='veld', values=([o[0] for o in vel_probs], [o[1] / unnormed_prob
                                                                                 for o in vel_probs]))
    return distrib


# In[89]:

def get_note(distr):
    return notes[distr.rvs()]


# In[90]:

def get_note_length(distr):
    return lengths_list[distr.rvs()]


# In[91]:

def get_volume(distr):
    return volumes_list[distr.rvs()]


# In[92]:

def write_to_midi(composition, comp_name, tempo=120):
    MyMIDI = MIDIFile(1)
    track = 0   
    time = 0
    channel = 0
    MyMIDI.addTrackName(track,time,"Composition")
    MyMIDI.addTempo(track, time, tempo)
    
    for note in composition:
        if note.value != 'silence':
            MyMIDI.addNote(track, channel, convert_pitch_to_MIDI(note.value, note.octave), note.start_time, note.duration,
                           note.volume)
                       
    binfile = open(comp_name + ".mid", 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()


# In[93]:

def compose_hand(n_d, vel_d, oct_d, nl_d, length, t0, chord=False, sequins=None, note_lengths=None, maxnotes_single=2,
                 maxnotes_chord=4, max_dist_single=10, no_dissonance=False, velvar=5,
                 doubling_dist=0):

    djshadow = []
    if chord:
        notes_chosen = 0
        note_names = []

        octave_chosen = oct_d.rvs()
        max_tries = 20
        tmp = 0
        while notes_chosen < maxnotes_chord - 1:  # at least 3 notes
            note = get_note(n_d)
            tmp += 1
            if tmp > max_tries:
                notes_chosen = maxnotes_chord - 1
            if note != 'silence' and note not in note_names:
                if notes_chosen == 0 or not no_dissonance:
                    notes_chosen += 1
                    note_names.append(note)
                    djshadow.append(Note(note, octave_chosen, t0, length,
                                            get_volume(vel_d) + get_vel_var(std=velvar)))
                else:
                    flag = False
                    for tmp_n in note_names:
                        if check_dissonance(note, tmp_n):
                            flag = True
                    if not flag:
                        notes_chosen += 1
                        note_names.append(note)
                        djshadow.append(Note(note, octave_chosen, t0, length,
                                                get_volume(vel_d) + get_vel_var(std=velvar)))

        note = get_note(n_d)  # a fourth note, perhaps?
        if note != 'silence' and note not in note_names:
            flag = False
            for tmp_n in note_names:
                if check_dissonance(note, tmp_n):
                    flag = True
            if not flag:
                djshadow.append(Note(note, octave_chosen, t0, length,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
    else:
        note_length_counter = 0
        if note_lengths is None:
            note_lengths = []
            while note_length_counter < length:
                nl = get_note_length(nl_d)
                if nl > length:
                    nl = length
                note_lengths.append(nl)
                note_length_counter += nl
        
        note_length_counter = 0
        prev_note = None
        for iii, nl in enumerate(note_lengths):
            chosen_notes = []
            
            if sequins is None:
                for j in range(maxnotes_single):  # no more than maxnotes_single
                    note = get_note(n_d)

                    if note != 'silence':
                        octave_chosen = oct_d.rvs()
                        
                        flag = True
                        while flag != False:
                            flag = False

                            for ntmp in chosen_notes:
                                if check_dissonance(note, ntmp[0]):
                                    flag = True
                                if abs(convert_pitch_to_MIDI(ntmp[0], ntmp[1])
                                       - convert_pitch_to_MIDI(note, octave_chosen)) > max_dist_single:
                                    flag = True
                        djshadow.append(Note(note, octave_chosen, t0 + note_length_counter, nl,
                                            get_volume(vel_d) + get_vel_var(std=velvar)))
                        chosen_notes.append([note, octave_chosen])
            else:
                if prev_note is None:
                    prev_note = [get_note(n_d), oct_d.rvs()]
                note = get_note(n_d)
                composito_flag = True
                if sequins[iii] == 'R':
                    note = prev_note[0]
                    octave_chosen = prev_note[1]
                if note == 'silence':
                    while note == 'silence':
                        note = get_note(n_d)
                    octave_chosen = oct_d.rvs()
                    composito_flag = False
                    prev_note = [note, octave_chosen]
                else:
                    octave_chosen = oct_d.rvs()
                    if sequins[iii] == 'R':
                        octave_chosen = prev_note[1]
                    flag = True
                    while flag != False:
                        flag = False
                        if prev_note is not None and note != 'silence' and prev_note[0] != 'silence':
                            if sequins[iii] == 'u':
                                if convert_pitch_to_MIDI(note, octave_chosen) < convert_pitch_to_MIDI(prev_note[0], prev_note[1]):
                                    flag = True
                            elif sequins[iii] == 'd':
                                if convert_pitch_to_MIDI(note, octave_chosen) > convert_pitch_to_MIDI(prev_note[0], prev_note[1]):
                                    flag = True
                        if flag:
                            note = get_note(n_d)
                            octave_chosen = oct_d.rvs()

                    if sequins[iii] == 'r':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                    elif sequins[iii] == 't2':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                        composito_flag = False
                        for i in range(2):
                            djshadow.append(Note(note, octave_chosen, t0 + note_length_counter + nl / 2 * i, nl / 2,
                                            get_volume(vel_d) + get_vel_var(std=velvar)))
                    elif sequins[iii] == 't3':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                        composito_flag = False
                        for i in range(2):
                            djshadow.append(Note(note, octave_chosen, t0 + note_length_counter + nl / 3 * i, nl / 3,
                                            get_volume(vel_d) + get_vel_var(std=velvar)))
                    elif sequins[iii] == 't4':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                        composito_flag = False
                        for i in range(2):
                            djshadow.append(Note(note, octave_chosen, t0 + note_length_counter + nl / 4 * i, nl / 4,
                                            get_volume(vel_d) + get_vel_var(std=velvar)))
                    elif sequins[iii] == 'ou':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                        composito_flag = False
                        djshadow.append(Note(note, octave_chosen + 1, t0 + note_length_counter, nl / 2,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                        djshadow.append(Note(note, octave_chosen, t0 + note_length_counter + nl / 2, nl / 2,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                    elif sequins[iii] == 'ou2':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                        composito_flag = False
                        djshadow.append(Note(note, octave_chosen + 2, t0 + note_length_counter, nl / 3,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                        djshadow.append(Note(note, octave_chosen + 1, t0 + note_length_counter + nl / 3, nl / 3,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                        djshadow.append(Note(note, octave_chosen, t0 + note_length_counter + 2 * nl / 3, nl / 3,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                    elif sequins[iii] == 'od':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                        composito_flag = False
                        djshadow.append(Note(note, octave_chosen - 1, t0 + note_length_counter, nl / 2,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                        djshadow.append(Note(note, octave_chosen, t0 + note_length_counter + nl / 2, nl / 2,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                    elif sequins[iii] == 'od2':
                        note = prev_note[0]
                        octave_chosen = prev_note[1]
                        composito_flag = False
                        djshadow.append(Note(note, octave_chosen - 2, t0 + note_length_counter, nl / 3,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                        djshadow.append(Note(note, octave_chosen - 1, t0 + note_length_counter + nl / 3, nl / 3,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                        djshadow.append(Note(note, octave_chosen, t0 + note_length_counter + 2 * nl / 3, nl / 3,
                                        get_volume(vel_d) + get_vel_var(std=velvar)))
                    if note == 'silence':
                        while note == 'silence':
                            note = get_note(n_d)
                        octave_chosen = oct_d.rvs()
                        composito_flag = False
                        prev_note = [note, octave_chosen]
                    if composito_flag:
                        djshadow.append(Note(note, octave_chosen, t0 + note_length_counter, nl,
                                            get_volume(vel_d) + get_vel_var(std=velvar)))
                    
                    prev_note = [note, octave_chosen]
            note_length_counter += nl
    return djshadow


# In[94]:
def main():
    parser = argparse.ArgumentParser(description='Instagram Crawler')
    parser.add_argument('--osc', type=int, default=0, help='If 0, will not send OSC messages, if 1, will send')
    parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5005, help="The port the OSC server is listening on")
    parser.add_argument("--address", type=str, default="/image", help="The address to which the OSC messages will be sent")

    args = parser.parse_args()


    if args.osc == 1:
        osc_client = udp_client.SimpleUDPClient(args.ip, args.port)
    else:
        osc_client = None

    if not isdir('MIDIoutput'):
        makedirs('MIDIoutput')
    maxname = 0
    for root, dirs, files in walk('MIDIoutput'):
        for name in files:
            if name.endswith('.mid'):
                if maxname < int(name[:-4]):
                    maxname = int(name[:-4])
                maxname += 1

    tonalities = [Bbmin, CMaj, Dmin, Chromatic, FSharp, FSharpHMinor, FSharpHMinor_noroot, Gmin, GSharpHMinor, Fmin]
    tonal_probs = 1./len(tonalities)
    seq_ton = stats.rv_discrete(name='tonalities', values=(list(range(len(tonalities))), [tonal_probs] * len(tonalities)))

    volume_probs_start = [[60, 0.4], [70, 0.3], [80, 0.25], [90, 0.05], [100., 0.2], [110, 0.1]]
    volume_probs_end = [[60, 0.1], [70, 0.2], [80, 0.45], [90, 0.25], [100., 0.], [110, 0.]]

    octave_probs_start_l = [[1, 0.2], [2, 0.3], [3, 0.25], [4, 0.25], [5, 0.], [6, 0.], [7, 0.]]
    octave_probs_end_l = [[1, 0.35], [2, 0.35], [3, 0.3], [4, 0.], [5, 0.], [6, 0.], [7, 0.]]

    octave_probs_start_r = [[1, 0.], [2, 0.], [3, 0.1], [4, 0.3], [5, 0.3], [6, 0.2], [7, 0.1]]
    octave_probs_end_r = [[1, 0.], [2, 0.], [3, 0.], [4, 0.2], [5, 0.4], [6, 0.25], [7, 0.15]]

    oct_probs = [[octave_probs_start_l, octave_probs_end_l], [octave_probs_start_r, octave_probs_end_r]]

    init_distribution_full_l = tonalities[seq_ton.rvs()]
    end_distribution_full_l = tonalities[seq_ton.rvs()]

    init_distribution_full_r = tonalities[seq_ton.rvs()]
    end_distribution_full_r = tonalities[seq_ton.rvs()]

    inter_d_l = tonalities[seq_ton.rvs()]
    inter_d_r = tonalities[seq_ton.rvs()]
    inter_oct_l = [[1, 0.3], [2, 0.35], [3, 0.2], [4, 0.15], [5, 0.], [6, 0.], [7, 0.]]
    inter_oct_r = [[1, 0.], [2, 0.], [3, 0.], [4, 0.15], [5, 0.15], [6, 0.4], [7, 0.3]]
    inter_pl = {'1/4': 0.35, '1/8': 0.35, '1/16': 0.025, '3/8': 0.15, '1/2': 0.1, '1': 0., '2': 0.025, '4': 0}
    inter_nl = {'1/4': 0.3, '1/8': 0.35, '1/16': 0.025, '3/8': .1, '1/2': .15, '1': .05, '2': 0.025, '4': 0.}
    inter_vol = [[60, 0.], [70, 0.], [80, 0.], [90, 0.4], [100., 0.3], [110, 0.3]]

    distrs = [[init_distribution_full_l, end_distribution_full_l], [init_distribution_full_r, end_distribution_full_r]]
    inter_distrs = [[inter_d_l, inter_d_l], [inter_d_r, inter_d_r]]
    inter_oct_probs = [[inter_oct_l, inter_oct_l], [inter_oct_r, inter_oct_r]]

    sequins_signes = ['u', 'd', 'r', 'R', 't2', 't3', 't4',
                      'ou', 'ou2', 'od', 'od2', ]  # up, down, repeat note, repeat even if silence intended,
                                                   # trills, octave skip from above
        
    piece_length_distr_start = {'1/4': 0.35, '1/8': 0., '1/16': 0., '3/8': 0.0, '1/2': 0.35, '1': 0.25, '2': 0.05, '4': 0}
    piece_length_distr_end = {'1/4': 0.35, '1/8': 0., '1/16': 0., '3/8': 0.0, '1/2': 0.35, '1': 0.25, '2': 0.05, '4': 0}

    note_length_distr_start = {'1/4': 0.4, '1/8': 0.25, '1/16': 0.05, '3/8': .15, '1/2': .15, '1': .0, '2': 0.0, '4': 0.}
    note_length_distr_end = {'1/4': 0.35, '1/8': 0.35, '1/16': 0.05, '3/8': .1, '1/2': .1, '1': .05, '2': 0.0, '4': 0.}

    sequins1 = ['R', 'R', 'R', 'R', 'R', 'R']
    sequins2 = ['d', 'd', 'd', 'd', 'd', 'd']
    sequins3 = ['t4', 't4', 't4', 't4', 't4', 't4']


    sequenco = [sequins1, sequins2, sequins3]

    # lengths = {'1/4': 1, '1/8': 0.5, '1/16': 0.25, '3/8': 1.5, '1/2': 2, '1': 4, '2': 8, '4': 16}
    nls1 = [0.5, 0.5, 0.5, 0.25, 0.25, 0.25]
    nls2 = [1, 2, 4, 2, 4, 2]
    nls3 = [1, 1, 0.5, 1, 0.5, 1]

    sequence_lengths = [nls1, nls2, nls3]


    # In[96]:
    no_dissonance = False
    velvar = 8
    max_dist_single = 20
    maxnotes_chord = 4
    chord_prob_start_l = 0.4
    chord_prob_end_l = 0.4
    chord_prob_start_r = 0.15
    chord_prob_end_r = 0.15
    sparsness_init = 0.7
    sparsness_final = 0.75
    sequence_prob = 0.3
    repetition_prob = 0.

    inter_no_diss = False
    inter_sparsness = 0.25
    inter_prob = 0.3
    inter_chord_p_l = 0.15
    inter_chord_p_r = 0.15
    inter_chord_probs = [[inter_chord_p_l, inter_chord_p_l], [inter_chord_p_r, inter_chord_p_r]]

    inter_d_l = tonalities[seq_ton.rvs()]
    inter_d_r = tonalities[seq_ton.rvs()]

    seq_d = stats.rv_discrete(name='seqd', values=([0, 1], [1-sequence_prob, sequence_prob]))
    rep_d = stats.rv_discrete(name='repd', values=([0, 1], [1-repetition_prob, repetition_prob]))

    chord_probs = [[chord_prob_start_l, chord_prob_end_l], [chord_prob_start_r, chord_prob_end_r]]


    # In[97]:

    max_t = 150
    max_t_rel = 200
    non_upd_amt = max_t_rel - max_t
    counter = 0
    composition = []
    rel_count = 0
    prev_piece_length = 0
    first_bar_done = False
    orig_no_diss = no_dissonance

    while counter < max_t_rel:
        no_dissonance = orig_no_diss
        doing_inter_flag = False
        vol_distr = get_vel_distr(volume_probs_start, volume_probs_end, rel_count, max_t)
        piece_length_distr = get_piece_length_distr(piece_length_distr_start, piece_length_distr_end, rel_count, max_t)
        note_length_distr = get_note_length_distr(note_length_distr_start, note_length_distr_end, rel_count, max_t)
        
        inter_pd = stats.rv_discrete(name='interd', values=([0, 1], [1. - inter_prob, inter_prob]))
        do_inter = inter_pd.rvs()
        
        if do_inter == 1 and max_t / 6 < counter < 6 * max_t / 7:
            doing_inter_flag = True
            vol_distr = get_vel_distr(inter_vol, inter_vol, rel_count, max_t)
            piece_length_distr = get_piece_length_distr(inter_pl, inter_pl, rel_count, max_t)
            note_length_distr = get_note_length_distr(inter_nl, inter_nl, rel_count, max_t)
            no_dissonance = inter_no_diss

        do_seq = seq_d.rvs()
        if do_seq == 1 and not doing_inter_flag:
            seq = sequenco[stats.randint.rvs(0, len(sequenco))]
            sql = sequence_lengths[stats.randint.rvs(0, len(sequence_lengths))]
            piece_length = sum(sql)
        else:
            seq = None
            sql = None

        for i in range(2):
            
            if doing_inter_flag:
                note_distr = get_note_distr(inter_distrs[i][0], inter_distrs[i][1], inter_sparsness, inter_sparsness,
                                            rel_count, max_t)
                octave_distr = get_octave_distr(inter_oct_probs[i][0], inter_oct_probs[i][0], rel_count, max_t)
                chord_distr = get_chord_single_distr(inter_chord_probs[i][0], inter_chord_probs[i][1], rel_count, max_t)
            else:
                note_distr = get_note_distr(distrs[i][0], distrs[i][1], sparsness_init, sparsness_final,
                                    rel_count, max_t)
                octave_distr = get_octave_distr(oct_probs[i][0], oct_probs[i][0], rel_count, max_t)
                chord_distr = get_chord_single_distr(chord_probs[i][0], chord_probs[i][1], rel_count, max_t)
            do_repeat = 0
            if first_bar_done:
                do_repeat = rep_d.rvs()

            if do_repeat == 1:
                for nn in prev_part[i]:
                    nn_n = nn
                    nn_n.start_time += prev_piece_length
                    composition.append(nn_n)
            else:
                if do_seq == 0:
                    piece_length = get_note_length(piece_length_distr)
            
                if chord_distr.rvs() == 0:  # we play a chord
                    partito = compose_hand(note_distr, vol_distr, octave_distr, note_length_distr, piece_length,
                                           counter, chord=True, sequins=None, note_lengths=None, maxnotes_single=2,
                                           maxnotes_chord=maxnotes_chord, max_dist_single=max_dist_single,
                                           no_dissonance=no_dissonance, velvar=velvar,
                                           doubling_dist=0)
                else:
                    if do_seq == 1:
                        partito = compose_hand(note_distr, vol_distr, octave_distr, note_length_distr, piece_length,
                                               counter, chord=False, sequins=seq, note_lengths=sql, maxnotes_single=1,
                                               maxnotes_chord=4, max_dist_single=max_dist_single,
                                               no_dissonance=False, velvar=velvar,
                                               doubling_dist=0)
                    else:
                        partito = compose_hand(note_distr, vol_distr, octave_distr, note_length_distr, piece_length,
                                               counter, chord=False, sequins=None, note_lengths=None, maxnotes_single=1,
                                               maxnotes_chord=4, max_dist_single=max_dist_single,
                                               no_dissonance=False, velvar=velvar,
                                               doubling_dist=0)
                if not first_bar_done:
                    first_bar_done = True
                    prev_part = [partito, partito]
                prev_part[i] = tuple(partito)
                prev_piece_length = piece_length
                for nn in partito:
                    composition.append(nn)


        counter += piece_length
        if non_upd_amt < counter < max_t + non_upd_amt:
            rel_count += piece_length        

    write_to_midi(composition, 'MIDIoutput/' + str(maxname))
    if osc_client is not None:
        osc_client.send_message(args.address, str(maxname) + '.mid')

if __name__ == "__main__":
    main()




