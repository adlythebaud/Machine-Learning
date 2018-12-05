import numpy as np

# 1. get movielens dataset
from lightfm.datasets import fetch_movielens

# 2. get LightFM class which will make a class for us.
from lightfm import LightFM

# 3. get data and format it
 
#lightfm will consider ratings that are 5 as positive and ratings that are 4 and below as negative to make this a binary problem.
#100K movie ratings from 1K users on 1700 movies
# Each user has rated at least 20 movies on a scale of 1-5
data = fetch_movielens(min_rating=4.0)

# 4. Create model. 
# We skipped feature scaling and all that good stuff.
# use a method called loss, which measures difference between model prediction and desired outocome

# warp helps us make recommendations by looking at user rating pairs and making rankings for each of those pairs.

# warp is content and collaborative
model = LightFM(loss ='warp')


# 5. Train model
model.fit(data['train'], epochs = 30, num_threads = 2)

n_users, n_items = data['train'].shape
    
    # get recommendations for each user we input
for user_id in range(200,203):
        
    # movies the user_id already likes. This is a compressed, sparse row format
    known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
    
    # movies our model predicts they will like
    scores = model.predict(user_id, np.arange(n_items))
    
    # rank them in order of most liked to least
    top_items = data['item_labels'][np.argsort(-scores)]
    print("User %s" % user_id)
    print("      Known positives:")
    
    for x in known_positives[:3]:
        print("           %s" % x)
    
    print("        Recommended:")
    
    for x in top_items[:3]:
        print("           %s" % x)

# 6. get a recommendation from model
# user_ids are the users we want to generate recommendations for.
def get_recommendation(model, data, user_ids):
    
    # get num users and num items
    n_users, n_items = data['train'].shape
    
    # get recommendations for each user we input
    for user_id in user_ids:
            
        # movies the user_id already likes. This is a compressed, sparse row format
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        print("User %s" % user_id)
        print("      Known positives:")
        
        for x in known_positives[:3]:
            print("           %s" % x)
        
        print("        Recommended:")
        
        for x in top_items[:3]:
            print("           %s" % x)
        


#get_recommendation(model, data, users)






    
