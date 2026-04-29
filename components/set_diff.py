#%%#
"""SetDiff core algorithm implementation"""
""" Import Dependencies """
import numpy as np
import json
from typing import Dict

from sklearn.preprocessing import StandardScaler

from serve.utils_clip import get_embeddings
from inverse_cca import InverseCCA

#%%#
class SetDiff:
    def __init__(self, args: Dict):
        self.args = args
    
    def pre_process(self, dataset):
        imgs = []
        cls_name = dataset[0]['group_name']
        for item in dataset:
            imgs.append(item['path'])
        return imgs, cls_name

    def dedup(self, diffs):
        diff_st = set()
        uniq_diffs = []
        for obj in diffs:
            if obj['text'] in diff_st:
                continue
            diff_st.add(obj['text'])
            uniq_diffs.append({'text':obj['text'], 'correlation':obj['correlation']})
        return uniq_diffs


    # considers both the frequency and similarity score in calculation
    def enhanced_frequency_filtering(self, class_img_embeds, universal_texts, universal_embeddings, top_k=10, similarity_threshold=0.50, min_threshold=0.25):
        """
        For each text embedding, compute its similarity to *all* class images,
        take the average similarity, then select top-k texts by that average.
        """

        # Normalize (cosine similarity via dot product)
        class_images_norm = class_img_embeds / (np.linalg.norm(class_img_embeds, axis=1, keepdims=True) + 1e-8)
        text_embeddings_norm = universal_embeddings / (np.linalg.norm(universal_embeddings, axis=1, keepdims=True) + 1e-8)

        # Similarity matrix: (num_images, num_texts)
        sim_matrix = class_images_norm @ text_embeddings_norm.T

        # Average similarity per text: (num_texts,)
        avg_sims = sim_matrix.mean(axis=0)

        # Pick top-k by average similarity
        k = min(top_k, len(universal_texts))
        top_indices = np.argsort(avg_sims)[-k:][::-1]

        filtered_texts = [{"text": universal_texts[i], "score": float(avg_sims[i])} for i in top_indices]
        filtered_embeddings = universal_embeddings[top_indices]

        return filtered_texts, filtered_embeddings


    def get_differences(self, class0_dataset, class1_dataset, seed):
        #extract images and text
        class0_imgs, cls0_name = self.pre_process(class0_dataset)
        class1_imgs, cls1_name = self.pre_process(class1_dataset)

        #extract embeddings
        class0_img_embeds = get_embeddings(
            class0_imgs, self.args["clip_model"], "image"
        )
        class1_img_embeds = get_embeddings(
            class1_imgs, self.args["clip_model"], "image"
        )
        knowledge_bank_filepath = self.args["knowledge_bank_filepath"]
        # Load universal vocabulary
        with open(knowledge_bank_filepath, 'r') as f:
            universal_data = json.load(f)
        universal_texts = list(set(universal_data))

        universal_text_embeddings = get_embeddings(
            universal_texts, self.args["clip_model"], "text"
        )

        """Filter vocabulary for each class"""
        class0_txts_objs, class0_txt_embeds = self.enhanced_frequency_filtering(
            class0_img_embeds, universal_texts, universal_text_embeddings, top_k=20, similarity_threshold=0.75
        )
        class0_txts = [obj['text'] for obj in class0_txts_objs]
        class0_txts_score_mp = {obj['text']:obj['score'] for obj in class0_txts_objs}
        class0_sim_scores = [obj['score'] for obj in class0_txts_objs]

        class1_txts_objs, class1_txt_embeds = self.enhanced_frequency_filtering(
            class1_img_embeds, universal_texts, universal_text_embeddings, top_k=20, similarity_threshold=0.75
        )
        class1_txts = [obj['text'] for obj in class1_txts_objs]
        class1_txts_score_mp = {obj['text']:obj['score'] for obj in class1_txts_objs}
        class1_sim_scores = [obj['score'] for obj in class1_txts_objs]

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

        alpha = 0.3
        inverse_cca_args = self.args["inverse_cca"]
        inverse_cca = InverseCCA(inverse_cca_args)
        # Analyze both mismatch cases
        cls0_vs_cls1, _ = inverse_cca.inverse_cca_analysis(
            class1_images_std, class0_texts_std, 
            class0_txts, class0_txt_embeds,
            scaler_txt_cls0, seed=seed
        )

        cls0_min_sim_score = min(class0_sim_scores)
        cls0_max_sim_score = max(class0_sim_scores)
        for obj in cls0_vs_cls1:
            txt = obj['text']
            anti_corr = 1.0 - abs(obj["correlation"])
            class0_txt_sim_score = class0_txts_score_mp[txt]
            class0_txt_sim_score_norm = (class0_txt_sim_score - cls0_min_sim_score) / (cls0_max_sim_score - cls0_min_sim_score + 1e-8)
            obj['sim_score'] = class0_txt_sim_score_norm
            obj['inv_corr_score'] = anti_corr
            obj['inv_diff_score'] = alpha*class0_txt_sim_score_norm + ((1-alpha)*anti_corr)

        cls1_vs_cls0, _ = inverse_cca.inverse_cca_analysis(
            class0_images_std, class1_texts_std,
            class1_txts, class1_txt_embeds,
            scaler_txt_cls1, seed=seed
        )

        cls1_min_sim_score = min(class1_sim_scores)
        cls1_max_sim_score = max(class1_sim_scores)
        for obj in cls1_vs_cls0:
            txt = obj['text']
            anti_corr = 1.0 - abs(obj["correlation"])
            class1_txt_sim_score = class1_txts_score_mp[txt]
            class1_txt_sim_score_norm = (class1_txt_sim_score - cls1_min_sim_score) / (cls1_max_sim_score - cls1_min_sim_score + 1e-8)
            obj['sim_score'] = class1_txt_sim_score_norm
            obj['inv_corr_score'] = anti_corr
            obj['inv_diff_score'] = alpha*class1_txt_sim_score_norm + ((1-alpha)*anti_corr)

        # Sort by absolute correlation (lowest first)
        cls0_vs_cls1.sort(key=lambda x: x['inv_diff_score'], reverse=True)
        cls1_vs_cls0.sort(key=lambda x: x['inv_diff_score'], reverse=True)

        final_keys = ['text', 'sim_score', 'inv_corr_score', 'inv_diff_score']
        cls0_diffs = [{
            key: obj.get(key) for key in final_keys
        } for obj in cls0_vs_cls1]
        cls1_diffs = [{
            key: obj.get(key) for key in final_keys
        } for obj in cls1_vs_cls0]

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
    