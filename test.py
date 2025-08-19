import t2v_metrics
video = './videos/baby.mp4'
text = 'a baby crying'
gem_score = t2v_metrics.VQAScore(model='qwen2.5-vl-7b-cambench')
print(gem_score(images=[video], texts=[text]))
