import scipy
import sounddevice as sd

from defs import project_root
from funcs import read_mp3

buzzer_sound = scipy.io.wavfile.read(project_root() / "buzzer.wav")
ok_sound = read_mp3(project_root() / "OK.mp3")
detect_sound = read_mp3(project_root() / "detect.mp3")


def play_detect_sound():
    sd.play(detect_sound[1], detect_sound[0])

def play_buzzer_sound():
    sd.play(buzzer_sound[1], buzzer_sound[0])

def play_ok_sound():
    sd.play(ok_sound[1], ok_sound[0])


if __name__ == '__main__':
    # play_detect_sound()
    # play_buzzer_sound()
    play_ok_sound()
    input()
