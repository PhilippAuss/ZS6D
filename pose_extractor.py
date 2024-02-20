import torch
from torch import nn
from torchvision import transforms
import extractor
from PIL import Image
from typing import Union, List, Tuple
from correspondences import chunk_cosine_sim
from sklearn.cluster import KMeans
import numpy as np
import time
from kmeans_pytorch import kmeans




class PoseViTExtractor(extractor.ViTExtractor):

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        self.model_type = model_type
        self.stride = stride
        self.model = model
        self.device = device
        super().__init__(model_type = self.model_type, stride = self.stride, model=self.model, device=self.device)

        self.prep = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=self.mean, std=self.std)
                         ])
        
    


    def preprocess(self, img: Image.Image, 
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        
        scale_factor = 1
        
        if load_size is not None:
            width, height = img.size # img has to be quadratic

            img = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img)

            scale_factor = img.size[0]/width

        
        prep_img = self.prep(img)[None, ...]

        return prep_img, img, scale_factor
    
    # Overwrite functionality of _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]
    # to extract multiple facets and layers in one turn
    
    def _extract_multi_features(self, batch: torch.Tensor, layers: List[int] = [9,11], facet: str = 'key') -> List[torch.Tensor]:
        B, C, H, W = batch.shape
        self._feats = []
        # for (layer,fac) in zip(layers,facet):
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats
    
    def extract_multi_descriptors(self, batch: torch.Tensor, layers: List[int] = [9,11], facet: str = 'key',
                        bin: List[bool] = [True, False], include_cls: List[bool] = [False, False]) -> torch.Tensor:

        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                        choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_multi_features(batch, layers, facet)
        descs = []
        for i, x in enumerate(self._feats):
            if facet[i] == 'token':
                x.unsqueeze_(dim=1) #Bx1xtxd
            if not include_cls[i]:
                x = x[:, :, 1:, :]  # remove cls token
            else:
                assert not bin[i], "bin = True and include_cls = True are not supported together, set one of them False."
            if not bin:
                desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
            else:
                desc = self._log_bin(x)
            descs.append(desc)
        return descs
            
    
    def find_correspondences_fastknn(self, pil_img1, pil_img2, prep_img1 = None, prep_img2 = None, desc_img1 = None, desc_img2 = None,
                                             num_pairs: int = 10, load_size: int = 224, 
                                             layer: int = 9, facet: str = 'key', bin: bool = True, 
                                             thresh: float = 0.05) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        
        start_time_corr = time.time()
        start_time_desc = time.time()
        
        if (desc_img1 is None) or (prep_img1 is None):
            image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
            descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
            
        else:
            image1_pil = pil_img1
            image1_batch = prep_img1
            descriptors1 = desc_img1
            
        num_patches1, load_size1 = self.num_patches, self.load_size
            
        if (desc_img2 is None) or (prep_img2 is None):
            image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
            descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
            
        else:
            image2_pil = pil_img2
            image2_batch = prep_img2
            descriptors2 = desc_img2
        
        num_patches2, load_size2 = self.num_patches, self.load_size
            

        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine
        
        
        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        # bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        # bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]
        # apply k-means on a concatenation of a pairs descriptors.
        # all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        all_keys_together = torch.cat((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = torch.sqrt((all_keys_together ** 2).sum(axis=1, keepdim=True))
        normalized = all_keys_together / length
        
        start_time_kmeans = time.time()
        #'euclidean'
        cluster_ids_x, cluster_centers = kmeans(X = normalized, num_clusters=n_clusters, distance='cosine', device=self.device)
        kmeans_labels = cluster_ids_x.detach().cpu().numpy()
        
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans
        
        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))
        
        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb
        
        elapsed_corr = end_time_corr - start_time_corr
        
        print(f"all_corr: {elapsed_corr}, desc: {elapsed_desc}, chunk cosine: {elapsed_time_chunk_cosine}, saliency: {elapsed_saliencey}, kmeans: {elapsed_kmeans}, bb: {elapsed_bb}")

        return points1, points2, image1_pil, image2_pil
    
    
    def find_correspondences_fastkmeans(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 224, 
                             layer: int = 9, facet: str = 'key', bin: bool = True, 
                             thresh: float = 0.05) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        
        start_time_corr = time.time()
        
        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine
        
        
        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        # bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        # bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        bb_descs1 = descriptors1[0, 0, bbs_mask, :]
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :]
        # apply k-means on a concatenation of a pairs descriptors.
        # all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        all_keys_together = torch.cat((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = torch.sqrt((all_keys_together ** 2).sum(axis=1, keepdim=True))
        normalized = all_keys_together / length
        
        start_time_kmeans = time.time()
        #'euclidean'
        cluster_ids_x, cluster_centers = kmeans(X = normalized, num_clusters=n_clusters, distance='cosine', device=self.device)
        kmeans_labels = cluster_ids_x.detach().cpu().numpy()
        
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans
        
        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans_labels, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))
        
        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb
        
        elapsed_corr = end_time_corr - start_time_corr
        
        #print(f"all_corr: {elapsed_corr}, desc: {elapsed_desc}, chunk cosine: {elapsed_time_chunk_cosine}, saliency: {elapsed_saliencey}, kmeans: {elapsed_kmeans}, bb: {elapsed_bb}")

        return points1, points2, image1_pil, image2_pil
    
    
    def find_correspondences(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 224, 
                             layer: int = 9, facet: str = 'key', bin: bool = True, 
                             thresh: float = 0.05) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        
        start_time_corr = time.time()
        
        start_time_desc = time.time()
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size
        end_time_desc = time.time()
        elapsed_desc = end_time_desc - start_time_desc

        start_time_saliency = time.time()
        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]
        end_time_saliency = time.time()
        elapsed_saliencey = end_time_saliency - start_time_saliency

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        start_time_chunk_cosine = time.time()
        similarities = chunk_cosine_sim(descriptors1, descriptors2)
        end_time_chunk_cosine = time.time()
        elapsed_time_chunk_cosine = end_time_chunk_cosine - start_time_chunk_cosine
        
        
        start_time_bb = time.time()
        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        # apply k-means on a concatenation of a pairs descriptors.
        all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
        normalized = all_keys_together / length
        
        start_time_kmeans = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        end_time_kmeans = time.time()
        elapsed_kmeans = end_time_kmeans - start_time_kmeans
        
        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))
        
        end_time_bb = time.time()
        end_time_corr = time.time()
        elapsed_bb = end_time_bb - start_time_bb
        
        elapsed_corr = end_time_corr - start_time_corr
        
        print(f"all_corr: {elapsed_corr}, desc: {elapsed_desc}, chunk cosine: {elapsed_time_chunk_cosine}, saliency: {elapsed_saliencey}, kmeans: {elapsed_kmeans}, bb: {elapsed_bb}")

        return points1, points2, image1_pil, image2_pil
    
    
    def find_correspondences_old(self, pil_img1, pil_img2, num_pairs: int = 10, load_size: int = 224, 
                             layer: int = 9, facet: str = 'key', bin: bool = True, 
                             thresh: float = 0.05) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
        
        image1_batch, image1_pil, scale_factor = self.preprocess(pil_img1, load_size)
        descriptors1 = self.extract_descriptors(image1_batch.to(self.device), layer, facet, bin)
        num_patches1, load_size1 = self.num_patches, self.load_size
        image2_batch, image2_pil, scale_factor = self.preprocess(pil_img2, load_size)
        descriptors2 = self.extract_descriptors(image2_batch.to(self.device), layer, facet, bin)
        num_patches2, load_size2 = self.num_patches, self.load_size

        # extracting saliency maps for each image
        saliency_map1 = self.extract_saliency_maps(image1_batch.to(self.device))[0]
        saliency_map2 = self.extract_saliency_maps(image2_batch.to(self.device))[0]

        # saliency_map1 = self.extract_saliency_maps(image1_batch)[0]
        # saliency_map2 = self.extract_saliency_maps(image2_batch)[0]
        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1 > thresh
        fg_mask2 = saliency_map2 > thresh

        # calculate similarity between image1 and image2 descriptors
        similarities = chunk_cosine_sim(descriptors1, descriptors2)

        # calculate best buddies
        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        bbs_mask = nn_2[nn_1] == image_idxs

        # remove best buddies where at least one descriptor is marked bg by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

        # applying k-means to extract k high quality well distributed correspondence pairs
        bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
        bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
        # apply k-means on a concatenation of a pairs descriptors.
        all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
        n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
        length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
        normalized = all_keys_together / length
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        bb_topk_sims = np.full((n_clusters), -np.inf)
        bb_indices_to_show = np.full((n_clusters), -np.inf)

        # rank pairs by their mean saliency value
        bb_cls_attn1 = saliency_map1[bbs_mask]
        bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
        bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
        ranks = bb_cls_attn

        for k in range(n_clusters):
            for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
                if rank > bb_topk_sims[label]:
                    bb_topk_sims[label] = rank
                    bb_indices_to_show[label] = i

        # get coordinates to show
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
            bb_indices_to_show]  # close bbs
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]
        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        return points1, points2, image1_pil, image2_pil





    

    

