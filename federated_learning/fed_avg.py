def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list

    parameters is a list containing param of each model. For example, [param_model1, param_model2,...]
    Your task is to take the average of param of the models and put that into dictionary new_params
    E.g: new_params = {'name': param,...}
    
    Hints:
    parameters[0].keys() includes names of the models. E.g: ['name1', 'name2',...]
    If you use for param in parameters, param is a dictionary. To get the weights of the model, use param[name].data
    """
    new_params = {}
    # START CODING HERE

    return new_params