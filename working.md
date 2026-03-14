Here’s the VisDiff pipeline, step-by-step, in short:
	1.	Input.
Two image sets: DA and DB. Goal: describe what is more true in DA than in DB.

⸻

Step 1: Sample images
	2.	Randomly sample a small subset from each set, typically 20 images from DA and 20 from DB (call them SA, SB).
They repeat this sampling 3 rounds to see more of the data.

⸻

Step 2: Caption images (BLIP-2)
	3.	Run BLIP-2 on every sampled image to get a short caption, e.g.
	•	“a group of jockeys and horses are racing…”
	•	“a person riding a horse in an arena…”

⸻

Step 3: Propose candidate differences (GPT-4)
	4.	Feed all captions from SA and SB into GPT-4 with a prompt: “Tell me concepts that are more true for Group A than Group B.”
	5.	GPT-4 returns ~10 candidate difference phrases, like:
	•	“horse racing events”
	•	“multiple jockeys”
	•	“people posing for a picture”
	6.	Do this for each sampling round and union all candidates into a set Y.

⸻

Step 4: Score each candidate with CLIP (Ranker)
	7.	Encode every image in DA and DB with CLIP’s image encoder → image feature e_x.
	8.	Encode each candidate text y with CLIP’s text encoder → text feature e_y.
	9.	For every image x and candidate y, compute similarity
v(x,y) = \cos(e_x, e_y).
	10.	For each candidate y, collect two score lists:
	•	scores on DA images
	•	scores on DB images
	11.	Compute a difference score s_y = AUROC of using v(x,y) to distinguish DA vs DB.
	•	High AUROC ⇒ that phrase is much more true for DA than DB.
	12.	Optionally run a t-test on the two score distributions and drop phrases that aren’t statistically significant.

⸻

Step 5: Output
	13.	Sort candidates by s_y and take top-k (e.g., top-5).
	14.	These top phrases are the final natural-language descriptions of how DA differs from DB (“people posing for a picture”, “horse racing events”, etc.).

That’s the core pipeline: sample → caption → propose with GPT-4 → score with CLIP → rank & output