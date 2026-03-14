#%%#
"""SetDiff Implementation"""
""" Import Dependencies """
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from serve.utils_clip import get_embeddings
import components.prompts as prompts
from serve.utils_llm import get_llm_output

#%%#
class SetDiff:
    def __init__(self, args: Dict):
        self.args = args
    
    def pre_process(self, dataset):
        imgs = []
        txts = []
        cls_name = dataset[0]['group_name']
        for item in dataset:
            imgs.append(item['path'])
            txts.append(item['caption'])
        return imgs, txts, cls_name

    def dedup(self, diffs):
        diff_st = set()
        uniq_diffs = []
        for obj in diffs:
            if obj['text'] in diff_st:
                continue
            diff_st.add(obj['text'])
            uniq_diffs.append({'text':obj['text'], 'correlation':obj['correlation']})
        return uniq_diffs


    def unstandardize_direction(self, dir_std, scaler_text):
        """
        Convert direction from standardized space back to raw space
        It rescales a CCA text direction back to the same space as the raw text embeddings, so you can compute cosine similarity against candidate text phrases fairly.
        """
        return dir_std / (scaler_text.scale_ + 1e-12)

    """SetDiff Approach"""
    def set_diff_analysis(self, images_std, texts_std, text_descriptions, text_embeddings_raw,
                        scaler_text, n_components=10, seed=0):
        """
        Proper implementation of SetDiff with correct pairing
        """
        Xs = images_std
        Ys = texts_std

        # 2) Choose a safe number of components
        max_nc = min(n_components, Xs.shape[1], Ys.shape[1], Xs.shape[0]-1, Ys.shape[0]-1)
        if max_nc < 1:
            return [], np.array([])

        # 3) Fit CCA
        cca = CCA(n_components=max_nc, scale=False)
        Xc, Yc = cca.fit_transform(Xs, Ys)

        # 4) Per-component correlations
        cors = np.array([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(max_nc)])
        order = np.argsort(np.abs(cors))  # SetDiff: smallest |corr| first

        # 5) Take least-correlated text directions, map to the same class texts
        Y_dirs_std = cca.y_rotations_[:, order]
        results = []
        
        for rank_k in range(min(20, Y_dirs_std.shape[1])):
            dir_std = Y_dirs_std[:, rank_k]
            # Map direction back to raw text space
            dir_raw = self.unstandardize_direction(dir_std, scaler_text)
            dir_raw = dir_raw / (np.linalg.norm(dir_raw) + 1e-12)

            # Find closest text description from the same class used in CCA
            # Use the raw text embeddings that were standardized for this class
            sims = cosine_similarity([dir_raw], text_embeddings_raw)[0]
            j = np.argmax(sims)
            comp_idx = order[rank_k]

            results.append({
                "component": comp_idx,
                "correlation": cors[comp_idx],
                "text": text_descriptions[j],
                "similarity": sims[j]
            })
        
        return results, cors

    def get_differences(self, class0_dataset, class1_dataset):
        #extract images and text
        class0_imgs, class0_txts, cls0_name = self.pre_process(class0_dataset)
        class1_imgs, class1_txts, cls1_name = self.pre_process(class1_dataset)

        #extract embeddings
        class0_img_embeds = get_embeddings(
            class0_imgs, self.args["clip_model"], "image"
        )
        class0_txt_embeds = get_embeddings(class0_txts, self.args["clip_model"], "text")

        class1_img_embeds = get_embeddings(
            class1_imgs, self.args["clip_model"], "image"
        )
        class1_txt_embeds = get_embeddings(class1_txts, self.args["clip_model"], "text")

        scaler_img_cls0 = StandardScaler()
        scaler_img_cls1 = StandardScaler()
        scaler_txt_cls0 = StandardScaler()
        scaler_txt_cls1 = StandardScaler()

        # Standardize image embeddings
        class0_images_std = scaler_img_cls0.fit_transform(class0_img_embeds)
        class1_images_std = scaler_img_cls1.fit_transform(class1_img_embeds)

        # Standardize text embeddings
        class0_texts_std = scaler_txt_cls0.fit_transform(class0_txt_embeds)
        class1_texts_std = scaler_txt_cls1.fit_transform(class1_txt_embeds)


        # Analyze both mismatch cases
        print(f"Analyzing {cls1_name} Images vs {cls0_name} Text...")
        cls1_vs_cls0, cors1 = self.set_diff_analysis(
            class1_images_std, class0_texts_std, 
            class0_txts, class0_txt_embeds,
            scaler_txt_cls0
        )

        print(f"Analyzing {cls0_name} Images vs {cls1_name} Text...")
        cls0_vs_cls1, cors2 = self.set_diff_analysis(
            class0_images_std, class1_texts_std,
            class1_txts, class1_txt_embeds,
            scaler_txt_cls1
        )

        # Sort by absolute correlation (lowest first)
        # print(f"\nFeatures from {cls0_name} texts that are distinctive to {cls0_name} (not seen in {cls1_name} images):")
        cls1_vs_cls0.sort(key=lambda x: abs(x['correlation']))
        # print(f"\nFeatures from {cls1_name} texts that are distinctive to {cls1_name} (not seen in {cls0_name} images):")
        cls0_vs_cls1.sort(key=lambda x: abs(x['correlation']))
        
        cls0_diffs = self.dedup(cls1_vs_cls0) # distinctive for cls0
        cls1_diffs = self.dedup(cls0_vs_cls1) # distinctive for cls1

        return cls0_diffs, cls0_name, cls1_diffs, cls1_name

    def get_hypotheses(
        self, class0_dataset, class1_dataset
    ) -> Tuple[List[str], Dict]:
        # cls0_name = class0_dataset[0]['group_name']
        # cls1_name = class1_dataset[0]['group_name']

        cls0_prompt = getattr(prompts, self.args["prompt_cls_0"])
        # cls1_prompt = getattr(prompts, self.args["prompt_cls_1"])

        captions0 = [
            f"Group A: {item['caption']}".replace("\n", " ").strip()
            for item in class0_dataset
        ]
        captions1 = [
            f"Group B: {item['caption']}".replace("\n", " ").strip()
            for item in class1_dataset
        ]
        caption_concat = "\n".join(captions0 + captions1)

        cls0_prompt = cls0_prompt.format(text=caption_concat)
        cls0_output = get_llm_output(cls0_prompt, self.args["model"])
        cls0_diffs = [line.replace("* ", "") for line in cls0_output.splitlines()]

        # cls1_prompt = cls1_prompt.format(text=caption_concat)
        # cls1_output = get_llm_output(cls1_prompt, self.args["model"])
        # cls1_diffs = [line.replace("* ", "") for line in cls1_output.splitlines()]
        # cls0_name, cls1_diffs, cls1_name
        logs = [{"prompt": cls0_prompt, "output": cls0_output}]
        return cls0_diffs, logs
    