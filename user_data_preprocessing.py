def preprocess_data_function():
    with open("model/user_code.txt", "r") as f:
        lines = f.readlines()
    user_code = "".join(lines)
    code = '''
def preprocess_data(features):

    # Start user code
{user_code}
    # End user code

    return features
'''.format(user_code=user_code)
    return code
