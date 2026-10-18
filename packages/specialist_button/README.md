# specialist-button

A talking button plays one stored recording when pressed. A marker frame is a
1.3 s signal stored in a button. A microphone that hears the frame learns which
button played it, the source id 0..35, and one event digit 0..5.

## Frame

| Part | Length | Content |
|---|---|---|
| Preamble | 250 ms + 250 ms | down chirp, then up chirp |
| d1, d2 | 2 × 100 ms each | source id = 6·d1 + d2 |
| event | 2 × 100 ms | meaning assigned by the application |
| check | 2 × 100 ms | (d1 + d2 + event) mod 6 |

The buttons reproduce two bands, L = 800–1150 Hz and H = 3250–3700 Hz. The up
chirp sweeps L and then H at one sweep rate; the down chirp is its time
reverse. Each band holds six sub-bands. Digit k is sent twice: a 100 ms sweep
across the middle 70% of sub-band k of L, then the same in H. The amplitude is
constant between a 5 ms raised-cosine ramp at each end.

Digits are coded by frequency, so echoes do not change them. The echo of a slot
lies in the other band from the next slot, and a digit that fades at one
frequency survives at the other.

## Program a button and hear it

Run from the repository root on a Mac with the built-in microphone and speakers
selected:

```
uv run button program 7 2       # store source id 7, event 2 in a button
uv run button playback          # print every frame the microphone hears
uv run button decode FILE.wav   # print every frame in a recording
```

**program.** A long press on the record key makes the button beep, and the
button stores sound from 0.57 s after the start of the beep. The pass listens
for the beep and plays the frame at −28 dBFS peak, 1.02 s after the start of
the beep, so the frame sits 0.45 s into the stored recording. The stored level
settles within that lead.

**playback.** Prints each frame as it ends: time, level, score, and source id
and event, or `unreadable` with its flags. A level bar follows the sound.
Ctrl-C ends the pass and prints, per source and event, the frame count and the
median level, score and clock error.

**decode.** Reads a WAV file of any length, 16 kHz or more; of several
channels, the first counts. It prints the same lines and summary as playback.

`program` and `playback` set the speaker and microphone to full volume and the
speaker to 0 when they end. Each saves its recording under `logs/button/`,
named by time stamp; `--out` picks another folder.

## Setup

| Rule | Basis |
|---|---|
| Place each microphone so the frames of its own button score 0.35 or more. | The preamble needs 0.25. Presses away from the microphone scored 0.25 to 0.45. |
| Keep every other button at least 9 dB below the own button at that microphone. | Simulated collisions: the own frame is read in 96% of trials at 9 dB and 100% at 12 dB; two frames within 3 dB both stay unread. |
| Program all buttons at one place in front of the Mac speaker. | The stored loudness follows the level at programming; one button played 9 dB apart after two programmings. |
| Record unprocessed audio as WAV at 16 kHz or more. | No automatic gain, noise suppression or voice mode. Only the Mac microphone is tested. |
| Keep the clock error of button and recorder together within ±0.2%. | The preamble search covers ±0.2%. Recorded buttons fitted from −0.2% to +0.15%. |

To check a setup, put the microphones and buttons in their final places and
press each button about ten times. Run `button playback`, or record with the
final recorder and run `button decode`. For each microphone, its own button
should give no unreadable frame, a median score of 0.35 or more, and a level at
least 9 dB above every other button.

## Decode in code

```python
from specialist_button import decode, generate_frame

frame = generate_frame(source_id=7, event=2)  # 48 kHz, peak 1
for d in decode(audio, fs, own_source_id=7):
  print(d.onset, d.source_id, d.event, d.score, d.flags, d.accepted)
```

`decode` correlates the recording with both preamble chirps over nine clock
errors from −0.2% to +0.2%. A frame starts where both correlations reach the
threshold one chirp apart. Each digit slot is read as the energy of each
sub-band divided by that sub-band's energy in the preamble, so the response of
button and room cancels. The digits are the valid codeword whose sub-bands hold
the largest share of the slot energy.

| Field | Meaning |
|---|---|
| `onset` | start of the frame from the start of `audio` |
| `source_id`, `event` | decoded values |
| `score` | the smaller of the two preamble correlations, 0..1 |
| `share` | share of the slot energy in the decoded sub-bands, 0..1 |
| `flags` | `ambiguous`: the best codeword is less than 1 dB above the second. `collision`: another preamble with other digits overlaps, at least 70% as strong |
| `accepted` | `source_id == own_source_id` and no flag |
| `clock_error` | clock error of the template that fitted |

A flagged frame's digits are a guess. Preambles within 0.4 s of a reported
frame count as its echoes, unless they carry other digits, no flag and a share
of 0.6 or more. `DecoderConfig` holds every threshold.

## Limits

- The recorded frames come from three buttons, sources 0, 1 and 7, and the Mac
  microphone.
- Two buttons of similar level pressed less than 20 ms apart can give a wrong
  unflagged frame: their preambles merge, so the collision flag cannot see
  them.
- In white noise, every frame is read down to 0 dB in-band SNR, and 82% to 88%
  at −5 dB.
