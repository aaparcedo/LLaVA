def get_model_name_from_path(model_path):
    model_path = model_path.lower()
    if 'clip' in model_path:
        if '336' in model_path:
            return 'clip336'
        else:
            return 'clip'
    elif 'llava' in model_path:
        if '1.5' in model_path:
            return 'llava1.5'
        else:
            return 'llava'
    elif 'blip2' in model_path:
        return 'blip2'