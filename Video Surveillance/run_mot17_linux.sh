for seq in data/MOT17/*/; do
  name=$(basename "$seq")
  python run.py --video "$seq" \
                --zones config/zones_template.json \
                --output results/mot17/$name/ \
                --mode sequence \
                --evaluate
done
