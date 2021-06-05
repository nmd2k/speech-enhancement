import subprocess
import os
# import ffmpeg
import moviepy.editor as mp

# src = './uploads'
# dst = './uploads'

# for root, dirs, filenames in os.walk(src, topdown=False):
#     #print(filenames)
#     for filename in filenames:
#         print('[INFO] 1',filename)
#         try:
#             _format = ''
#             # if ".flv" in filename.lower():
#             #     _format=".flv"
#             # if ".mp4" in filename.lower():
#             #     _format=".mp4"
#             # if ".avi" in filename.lower():
#             #     _format=".avi"
#             if ".mov" in filename.lower():
#                 _format=".mov"
#             if _format == ".mov":
#                 inputfile = os.path.join(root, filename)
#                 print('[INFO] 1',inputfile)
#                 outputfile = os.path.join(dst, filename.lower().replace(_format, ".mp4"))
#                 subprocess.call(['ffmpeg', '-i', inputfile, outputfile])  
#         except:
#             print("not mov")

audio = './uploads/alarm.ogg'
video = './uploads/alarm.mp4'
read = open('./uploads/alarm.mp4', 'rb')
out = './out.mov'

# subprocess.call(['ffmpeg', '-i', '-i', video, audio, out])
# ffmpeg -i $video -i $audio -acodec copy -vcodec copy -map 0:v:0 -map 1:a:0 $out
vid =  mp.VideoFileClip("alarm.mp4")

background_music = mp.AudioFileClip("alarm.mp3")

# vid.audio = background_music
new_clip = vid.audio.set_audio(mp.AudioFileClip("alarm.mp3"))

new_clip.write_videofile("final_cut.mp4")
