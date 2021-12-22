def assign_models_priority(user_constraints, models):

    filtered_models = []
    if user_constraints['is_real_time']:
        for model in models:
            if model['inference_rate'] >= 30:
                filtered_models.append(model)


    if user_constraints['minimum_efectiveness'] is not None:
        for model in models:
            if model['efectiveness'] >= user_constraints['minimum_efectiveness']:
                filtered_models.append(model)


    if len(filtered_models) != 0:
        working_models = filtered_models
    else:
        working_models = models


    sorted_models = sorted(working_models, key=lambda x: (x['efectiveness'], x['training_rate']), reverse=True)

    final_models = []
    for i, model in enumerate(sorted_models):
        model['priority'] = i
        final_models.append(model)
        
    return final_models





