def assign_models_priority(user_constraints, models):

    filtered_models = []
    if user_constraints.get('is_real_time'):
        for model in models:
            if model['inference_rate'] >= 30:
                filtered_models.append(model)


    if user_constraints.get('minimum_efectiveness') is not None:
        for model in models:
            if model['efectiveness'] >= user_constraints['minimum_efectiveness']:
                filtered_models.append(model)


    if len(filtered_models) != 0:
        working_models = filtered_models
    else:
        working_models = models

    for model in working_models:
        if model.get("priority_weight") is None:
            model["priority_weight"] = 0

        if model.get("efectiveness") is None:
            model["efectiveness"] = -1
        
        if model.get("training_rate") is None:
            model["training_rate"] = -1


    sorted_models = sorted(working_models, key=lambda x: (x['priority_weight'], x['efectiveness'], x['training_rate']), reverse=True)

    final_models = []
    for i, model in enumerate(sorted_models):
        model['priority'] = i
        final_models.append(model)
        
    return final_models





