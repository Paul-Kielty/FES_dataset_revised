### A revised list of annotations for the [FES dataset](https://www.mdpi.com/2687372), repairing/excluding samples based on a number of logical checks and a manual review. 

To be used with the **raw** FES videos, not the preprocessed versions. 
More information and experimental results can be found in our paper: **[Locally Adaptive Decay Surfaces for High-Speed Face and Landmark Detection with Event Cameras](https://arxiv.org/abs/2602.23101)**. Citation:
``` 
@misc{kielty2026locallyadaptivedecaysurfaces,
      title={Locally Adaptive Decay Surfaces for High-Speed Face and Landmark Detection with Event Cameras}, 
      author={Paul Kielty and Timothy Hanley and Peter Corcoran},
      year={2026},
      url={https://arxiv.org/abs/2602.23101}, 
}
```
___

#### The checks used to screen the annotations are listed below and the implementation given in [FESDatasetLabelCheck.py](https://github.com/Paul-Kielty/FES_dataset_revised/blob/main/FESDatasetLabelCheck.py).

- **Annotation count per-timestamp:** Exclude samples found to have multiple label entries for a single timestamp (when only one subject is present in the video).

- **Landmark count:** Exclude annotations with fewer than 5 landmark positions.

- **Landmarks within bounding box:** The landmark positions were checked to ensure they are within the face bounding box. However, because correct landmarks were sometimes excluded due to a bounding box that was slightly too small, an additional repair check was implemented: if landmarks were outside the bounding box by less than 10\% of the diagonal length, the box dimensions were extended to exactly include the landmarks. Otherwise, the sample was discarded.

- **Facial topology:** To detect instances of inconsistent landmark indexing, a set of rules were created to validate the structure of the face:
    - First, ensure that the left mouth and eye landmarks are actually positioned to the left of the corresponding right landmark. 
    - A similar check compared the vertical positions of each eye landmark to the mouth landmark on the same side.
    - Verification of the nose position first required gathering some information of the head pose by comparing vertical and horizontal distances between facial feature pairs. When the is primarily turned to one side (yaw) - determined by the vertical distances dominating the horizontal - the nose landmark is required to have a vertical position between the average eye level and average mouth level. When the head is primarily facing up or down (pitch) - detected by dominant horizontal differences - the horizontal component of nose landmark must be fall between the left-side and right-side facial landmarks.

- **Spatiotemporal consistency:** This check ensures that facial landmarks and bounding boxes move together cohesively to catch annotation errors where landmarks are frozen or lag behind face movement. Inconsistencies between landmark and bounding box movements are detected across consecutive labels by computing displacement vectors for both elements. Samples are flagged where the absolute difference between the landmark displacement and bounding box displacement exceeds 20\% of the bounding box diagonal.