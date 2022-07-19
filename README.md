# Explore Number of Feature in Random Forest
#import packages from library

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
        from numpy import mean, std
        import matplotlib.pyplot as plt

#explore number of features 
#get a list of models to evaluate

        def get_models():
            models= dict()
             # explore number of features from 1 to 20
            for i in range (1,21):
                models[str(i)]= RandomForestClassifier(max_features=i, random_state=42, class_weight='balanced')
            return models

#evaluate a given model using cross-validation
        def evaluate_model (model, X, y):
            # define the evaluation procedure
            cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model and collect the results
            scores=cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy')
            return scores

#get the models to evaluate
        models= get_models()
#evaluate the models and store results
        results, names =list(), list()
        for name, model in models.items():
            # evaluate the model
            scores= evaluate_model(model, X, y)
            # store the results
            results.append(scores)
            names.append(name)
             # summarize the performance along the way
            print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
#plot model performance for comparison
        plt.boxplot(results, labels=names, showmeans=True)
        plt.show()

