# Model-selection-and-learning-for-nutritional-data
Model selection and learning pipeline for nutritional data, linking macronutrients intake to the prediction of several clinical outcomes.

**Publication** : https://doi.org/10.3390/nu13041377

**Abstract** : Although lifestyle-based interventions are the most effective to prevent metabolic syndrome
(MetS), there is no definitive agreement on which nutritional approach is the best. The aim of the
present retrospective analysis was to identify a multivariate model linking energy and macronutrient
intake to the clinical features of MetS. Volunteers at risk of MetS (F = 77, M = 80) were recruited in
four European centres and finally eligible for analysis. For each subject, the daily energy and nutrient
intake was estimated using the EPIC questionnaire and a 24-h dietary recall, and it was compared
with the dietary reference values. Then we built a predictive model for a set of clinical outcomes
computing shifts from recommended intake thresholds. The use of the ridge regression, which
optimises prediction performances while retaining information about the role of all the nutritional
variables, allowed us to assess if a clinical outcome was manly dependent on a single nutritional
variable, or if its prediction was characterised by more complex interactions between the variables.
The model appeared suitable for shedding light on the complexity of nutritional variables, which
effects could be not evident with univariate analysis and must be considered in the framework of the
reciprocal influence of the other variables.

**Code**: A sample pipeline built over sklearn objects, to perform model selection and fine tuning over the learnt best model using different multivariate regressions.
Some plotting utitlities are also implemented: spearman heatmap correlation plotting, R score prediction plotting with normalized gradients for the features and Ridge coefficients plotting. The latter is useful for visualizing strong monovariate effects of inadequate macronutrient intake on the clinical targets.


![Sample fine tuned prediction using Ridge Regression; gradient describes protein intake](https://github.com/CarloMengucci/Model-selection-and-learning-for-nutritional-data/blob/main/delta_grads_m_BMI_Protein_delta.png)


*Sample fine tuned prediction using Ridge Regression; gradient describes protein intake*

**Requirements**: code is compatible with Python 3.8 environments, requiring the following packages to be properly installed: Numpy, Pandas, SKlearn, Seaborn
