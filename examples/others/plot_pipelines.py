"""
Exploiting Pipelines!
=====================

This example....

"""


#######################################
# How pipelines should look like
# ------------------------------

# Create steps
#step1 = Trim(start=5, end=5)
#step2 = Unchanged(param1=x, param2=y)
#step3 = LostConnection(param1=20, param2=30)

# .. note: Similar to scikits pipeline which basically loops over
#          steps and executes for each step the methods .fit and
#          .transform.

# Create pipeline
#pipe = Pipeline(steps=[('step1', step1),
#                       ('step2', step2),
#                       ('step3', step3))

# Data
#data = pipe.fit_transform(data)
