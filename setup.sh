cp -r ./salad/salad /opt/conda/envs/salad

apt-get update
apt-get install ffmpeg
rm /opt/conda/bin/ffmpeg
ln -s /usr/bin/ffmpeg /opt/conda/bin/ffmpeg