# Gaze Direction Detection & Annotation
This repository contains code accompanying my Bachelor's thesis [“Dominance is relative: Automatic gaze annotation and conversational dominance in the MULTISIMO corpus”](/resources/thesis.pdf).

Two papers building on this work are currently under review for publication at IEEE CogInfoCom 2020:
	1. L. McLaren, M. Koutsombogera, and C. Vogel, “A heuristic method for automatic gaze detection in constrained multi-modal dialogue corpora”
	2. L. McLaren, M. Koutsombogera, and C. Vogel, “Gaze, dominance and dialogue role in the MULTSIMO
	corpus”

One aim of this project was to automatically generate annotations for gaze direction to be imported into an annotation tool, ELAN. Comparison of the results of automatic annotation to those of a human annotator resulted in a modified Kappa score of 0.79, indicating good inter-annotator agreement.

Consequent analysis of the resulting gaze annotation data is discussed in detail in my thesis, linked above.

MULTISIMO is a multimodal, multiparty dialogue corpus developed in Trinity College Dublin to enable the study of “human social behaviour in multiparty interaction, the structural representation of the interaction flow, and the modelling of the social communication mechanism to be integrated into intelligent collaborative systems”[^1]. Further information and access to the corpus data can be found [here](http://multisimo.eu).

***

*video_analysis.py* -- Performs frame-by-frame analysis of video, generating an ordinal direction (i.e. left, right, centre) based on iris position as well as a gaze direction vector based on head orientation. 

![Sample of Analysis](/resources/sample_frame.png "Sample of Analysis")

*shape_predictor_68_face_landmarks.dat* -- A pre-trained Dlib model that enables the location of facial landmarks.

*fine_annotation.py* -- Creates fine-grained annotations which can be imported into ELAN, from the output of *video_analysis.py*

*broad_annotation.py* -- Creates broad annotations which can be imported into ELAN, from the output of *fine_annotation.py*

### Sample broad annotation output 
|Begin Time | End Time | Direction         | Duration |
|-----------|----------|-------------------|----------|
| 0         | 233      | Gaze_Player-Left  | 233      |
| 233       | 700      | Gaze_Away         | 467      |
| 700       | 1534     | Gaze_Player-Left  | 834      |
| 1534      | 2602     | Gaze_Away         | 1067     |
| 2602      | 30463    | Gaze_Player-Right | 27861    |


[^1]: M. Koutsombogera and C. Vogel, “The MULTSIMO multimodal corpus of collaborative interactions,” in *Proceedings of the 19th ACM International Conference on Multimodal Interaction*, 2019, pp. 502 – 503.
