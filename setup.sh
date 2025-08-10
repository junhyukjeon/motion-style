# cp -r ./salad/salad /opt/conda/envs/salad
cp -r ./env /opt/conda/envs/env

apt-get update
apt-get install ffmpeg
rm /opt/conda/bin/ffmpeg
ln -s /usr/bin/ffmpeg /opt/conda/bin/ffmpeg

echo 'export PATH=/source/junhyuk/blender:$PATH' >> ~/.bashrc