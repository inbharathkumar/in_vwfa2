##### Stage 2 ROIs
The idea of generating the Stage 2 ROI plots by applying individual factors is to visually analyze how the different factors contribute to the final Stage 2 ROIs. So, the scoring logic wouldn't change, but instead of generating the Stage 2 ROI plots after applying all the scores, it will be generated at every level of analysis.

Individual Factors
1. Parcellation
2. Nontarget Contrast (Faces and Limb)
3. Session Count
4. Cortical Surface - (Sulcal Depth + Thickness + Curvature)

5a. Nontarget Contrast - Faces only
5b. Nontarget Contrast - Limbs only
6a. Cortical Surface - Sulcal Depth only
6b. Cortical Surface - Thickness only
6c. Cortical Surface - Curvature only

Paired Combinations
1&2, 1&3, 1&4, 2&3, 2&4, 3&4
