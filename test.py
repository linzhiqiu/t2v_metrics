import t2v_metrics
video = './videos/baby.mp4'
text = 'a baby crying'
gem_score = t2v_metrics.VQAScore(model='gemini-1.5-flash', api_key='AIzaSyDI4FXAxtnw5IqB7NJoR7TYKPdHGGffKvg')
print(gem_score(images=[video], texts=[text]))
