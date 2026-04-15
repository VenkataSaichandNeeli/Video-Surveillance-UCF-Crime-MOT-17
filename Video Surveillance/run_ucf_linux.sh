# Linux / macOS
for f in data/ucf/Fighting/*.mp4; do
  name=$(basename "$f" .mp4)
  python run.py --video "$f" \
                --zones config/zones_template.json \
                --output results/ucf_fighting/$name/
done
