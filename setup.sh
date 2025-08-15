# cp -r ./salad/salad /opt/conda/envs/salad
cp -r ./style /opt/conda/envs/style

apt-get update
apt-get install ffmpeg
rm /opt/conda/bin/ffmpeg
ln -s /usr/bin/ffmpeg /opt/conda/bin/ffmpeg

echo 'export PATH=/source/junhyuk/blender:$PATH' >> ~/.bashrc