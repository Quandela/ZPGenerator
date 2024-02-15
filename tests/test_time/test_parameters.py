from zpgenerator.time.parameters.parameterized_object import *
from zpgenerator.time.parameters.dictionary import Parameters

d = Parameters.DELIMITER


def test_child_parameters():
    parameters = [ParameterizedObject(parameters={'a': 1})]
    children = ChildParameterizedObject(children=parameters)
    assert len(children.children) == 1
    assert [child.parameters for child in children] == [['a']]
    parameters.append(ParameterizedObject(parameters={'b': 2}))
    assert len(children.children) == 2
    assert [child.parameters for child in children] == [['a'], ['b']]
    parameters[0].add_child(ParameterizedObject(parameters={'c': 3}))
    assert [child.parameters for child in children] == [['a', 'c'], ['b']]


def test_init_empty():
    par = ParameterizedObject()
    assert par._default_parameters == {}
    assert par.name is None
    assert par.default_name == '_ParameterizedObject'
    assert par._children.children == []
    assert par._keys == []
    assert par._pass_parameters is True


def test_init_full():
    par = ParameterizedObject(parameters={'parameter': 0}, name='name')
    assert par._default_parameters == {'parameter': 0}
    assert par.name == 'name'
    assert par._name == 'name'
    assert par._children.children == []
    assert par.keys == ['parameter']
    assert par._pass_parameters is False


def test_set_parameter():
    alice = ParameterizedObject(parameters={'age': 27}, name='Alice')

    assert alice.set_parameters().user == {}
    assert alice.set_parameters().default == {'age': 27}

    assert alice.get_parameters() == {'age': 27}

    assert alice.set_parameters({'age': 25}).user == {'age': 25}
    assert alice.set_parameters({'age': 25}).default == {'age': 27}
    assert alice.set_parameters({'age': 25}).dict == {'age': 25}

    assert alice.set_parameters({'Alice/age': 25}).user == {'age': 25}
    assert alice.set_parameters({'*/age': 25}).user == {'age': 25}
    assert alice.set_parameters({'height': 165}).default == {'age': 27}
    assert alice.set_parameters({'height': 165}).user == {}
    assert alice.set_parameters({'age': 22, '_age': 26}).default == {'age': 26}
    assert alice.set_parameters({'age': 22, '_age': 26}).user == {'age': 22}
    assert alice.set_parameters({'age': 22, '_age': 26}).dict == {'age': 22}


def test_children():
    smith = ParameterizedObject(name='Smith')
    alice = ParameterizedObject(parameters={'age': 27}, name='Alice')
    bob = ParameterizedObject(parameters={'age': 25, 'height': 176}, name='Bob')
    smith.add_child(alice)
    smith.add_child(bob)
    assert bob.set_parameters({'age': 23}).user == {'age': 23}
    assert bob.set_parameters({'age': 23}).default == {'age': 25, 'height': 176}
    assert smith._default_parameters == {}
    assert smith.default_parameters == {d.join(['Alice', 'age']): 27,
                                        d.join(['Bob', 'age']): 25,
                                        d.join(['Bob', 'height']): 176}
    assert smith.parameters == [d.join(names) for names in [['Alice', 'age'],
                                                            ['Bob', 'age'],
                                                            ['Bob', 'height']]]
    assert smith.parameter_tree() == {'Alice': {'age': 27}, 'Bob': {'age': 25, 'height': 176}}
    assert smith.parameter_tree({'Alice' + d + 'age': 35}) == {'Alice': {'age': 35}, 'Bob': {'age': 25, 'height': 176}}

    smith_set = smith.set_parameters({'Alice' + d + 'age': 35})
    assert smith_set.user == {'Alice' + d + 'age': 35}
    assert smith_set.default == {}
    assert alice.set_parameters(smith_set).user == {'age': 35}
    assert alice.set_parameters(smith_set).default == {'age': 27}
    assert alice.set_parameters(smith_set).dict == {'age': 35}

    assert bob.set_parameters(smith_set).user == {}
    assert bob.set_parameters(smith_set).default == {'age': 25, 'height': 176}
    assert bob.set_parameters(smith_set).dict == {'age': 25, 'height': 176}

    smith_set = smith.set_parameters({'age': 35})
    assert smith_set.user == {'age': 35}
    assert alice.set_parameters(smith_set).dict == {'age': 35}
    assert bob.set_parameters(smith_set).dict == {'age': 35, 'height': 176}


def test_children_overwrite():
    smith = ParameterizedObject(name='Smith')
    alice = ParameterizedObject(parameters={'age': 27, 'weight': 70}, name='Bob')
    bob = ParameterizedObject(parameters={'age': 25, 'height': 176}, name='Bob')
    smith.add_child(alice)
    smith.add_child(bob)
    assert smith._default_parameters == {}
    assert smith.parameter_tree() == {'Bob': {'weight': 70, 'age': 27}, 'Bob (1)': {'age': 25, 'height': 176}}
    assert smith.default_parameters == {d.join(['Bob', 'weight']): 70,
                                        d.join(['Bob', 'age']): 25,
                                        d.join(['Bob', 'height']): 176}
    assert smith.parameters == [d.join(names) for names in [['Bob', 'age'],
                                                            ['Bob', 'height'],
                                                            ['Bob', 'weight']]]


def test_rename():
    smith = ParameterizedObject(name='Smith')
    alice = ParameterizedObject(parameters={'age': 27, 'weight': 70}, name='Alice')
    bob = ParameterizedObject(parameters={'age': 25, 'height': 176}, name='Bob')
    smith.add_child(alice)
    smith.add_child(bob)
    assert smith.parameters == ['Alice' + d + 'age', 'Alice' + d + 'weight', 'Bob' + d + 'age', 'Bob' + d + 'height']
    assert smith.set_parameters().default == {}

    smith.rename_parameter('Bob' + d + 'age', 'Bob' + d + 'birthdays')
    assert smith.parameters == ['Alice' + d + 'age',
                                'Alice' + d + 'weight',
                                'Bob' + d + 'birthdays',
                                'Bob' + d + 'height']
    assert smith.set_parameters(parameters={'Bob' + d + 'birthdays': 22}).default == {}
    assert smith.set_parameters(parameters={'Bob' + d + 'birthdays': 22}).user == {'Bob' + d + 'age': 22}
    assert bob.set_parameters(smith.set_parameters(parameters={'Bob' + d + 'birthdays': 22})).default == {'age': 25,
                                                                                                          'height': 176}
    assert bob.set_parameters(smith.set_parameters(parameters={'Bob' + d + 'birthdays': 22})).user == {'age': 22}

    smith.rename_parameter('Bob' + d + 'birthdays', 'Bob' + d + 'times around the sun')
    assert smith.parameters == ['Alice' + d + 'age',
                                'Alice' + d + 'weight',
                                'Bob' + d + 'height',
                                'Bob' + d + 'times around the sun']
    bd = 'times around the sun'
    assert smith.set_parameters(parameters={'Bob' + d + bd: 22}).default == {}
    assert smith.set_parameters(parameters={'Bob' + d + bd: 22}).user == {'Bob' + d + 'age': 22}
    assert bob.set_parameters(smith.set_parameters(parameters={'Bob' + d + bd: 22})).default == {'age': 25,
                                                                                                 'height': 176}
    assert bob.set_parameters(smith.set_parameters(parameters={'Bob' + d + bd: 22})).user == {'age': 22}

    smith.rename_parameter('Alice' + d + 'weight', 'Alice' + d + 'heaviness')
    assert smith.parameters == ['Alice' + d + 'age', 'Alice' + d + 'heaviness', 'Bob' + d + 'height',
                                'Bob' + d + 'times around the sun']
    assert smith.default_parameters == {'Alice' + d + 'age': 27, 'Alice' + d + 'heaviness': 70,
                                        'Bob' + d + 'height': 176, 'Bob' + d + 'times around the sun': 25}
    assert smith.local_default_parameters == {}

    smith.update_default_parameters({'Alice' + d + 'heaviness': 66})
    assert smith.local_default_parameters == {'Alice' + d + 'heaviness': 66}
    assert smith.default_parameters == {'Alice' + d + 'age': 27, 'Alice' + d + 'heaviness': 66,
                                        'Bob' + d + 'height': 176, 'Bob' + d + 'times around the sun': 25}

    assert smith._renames == {'Bob' + d + 'age': 'Bob' + d + 'times around the sun',
                              'Alice' + d + 'weight': 'Alice' + d + 'heaviness'}

    assert smith.uses_parameter('Alice' + d + 'heaviness')
    assert smith.set_parameters({'Smith' + d + 'Alice' + d + 'heaviness': 50}).dict == {'Alice/weight': 50}

    assert alice.set_parameters(smith.set_parameters({'Smith' + d + 'Alice' + d + 'heaviness': 50})).dict == \
           {'age': 27, 'weight': 50}

    assert smith.set_parameters({'Smith' + d + '*' + d + 'age': 55}).dict == {'*/age': 55, 'Alice/weight': 66}
    assert alice.set_parameters(smith.set_parameters({'Smith' + d + '*' + d + 'age': 55})).dict == {'age': 55,
                                                                                                    'weight': 66}
    assert bob.set_parameters(smith.set_parameters({'Smith' + d + '*' + d + 'age': 55})).dict == {'age': 55,
                                                                                                  'height': 176}


def test_insert_parameter_function():
    obj = ParameterizedObject(parameters={'age': 27, 'weight': 70, 'height': 180})
    obj.add_child(ParameterizedObject(parameters={'bmi': 20}))
    assert obj.default_parameters == {'age': 27, 'bmi': 20, 'height': 180, 'weight': 70}
    assert obj.local_default_parameters == {'age': 27, 'height': 180, 'weight': 70}
    assert obj._output_local_default_parameters == {'age': 27, 'height': 180, 'weight': 70}

    obj.create_insert_parameter_function(lambda args: {'bmi': args['weight'] / (args['height'] / 100) ** 2},
                                         parameters={'weight': 80, 'height': 170})

    assert obj.local_default_parameters == {'age': 27, 'height': 170, 'weight': 80}
    assert obj._output_local_default_parameters == {'age': 27, 'bmi': 27.68166089965398, 'height': 170, 'weight': 80}

    assert obj.default_parameters == {'age': 27, 'bmi': 27.68166089965398, 'height': 170, 'weight': 80}

    assert obj.set_parameters().user == {}
    assert obj.set_parameters().dict == {'age': 27, 'bmi': 27.68166089965398, 'height': 170, 'weight': 80}

    assert obj.set_parameters({'bmi': 20}).user == {}
    assert obj.set_parameters({'bmi': 20}).dict == {'age': 27, 'bmi': 27.68166089965398, 'height': 170, 'weight': 80}

    assert obj.set_parameters({'height': 160}).user == {'bmi': 31.249999999999993, 'height': 160}
    assert obj.set_parameters({'height': 160}).default == {'age': 27, 'weight': 80}


def test_default_parameter_function():
    obj = ParameterizedObject(parameters={'age': 27, 'weight': 70, 'height': 180})
    obj.add_child(ParameterizedObject(parameters={'bmi': 20}))
    assert obj.default_parameters == {'age': 27, 'bmi': 20, 'height': 180, 'weight': 70}
    obj.create_default_parameter_function(lambda args: {'bmi': args['weight'] / (args['height'] / 100) ** 2},
                                          parameters={'weight': 70, 'height': 180})
    assert obj.default_parameters == {'age': 27, 'bmi': 21.604938271604937, 'height': 180, 'weight': 70}

    assert obj.set_parameters({'bmi': 20}).default == {'age': 27, 'height': 180, 'weight': 70, 'bmi': 20}
    assert obj.set_parameters({'bmi': 20}).user == {}

    assert obj.set_parameters({'height': 160}).default == {'age': 27, 'weight': 70}
    assert obj.set_parameters({'height': 160}).user == {'height': 160, 'bmi': 27.343749999999996}


def test_overwrite_parameter_function():
    obj = ParameterizedObject(parameters={'age': 27, 'weight': 70, 'height': 180})
    obj.add_child(ParameterizedObject(parameters={'current_year': 2020, 'birth_year': 1991}))
    obj.create_overwrite_parameter_function(lambda args: {'age': args['current_year'] - args['birth_year']},
                                            parameters={'current_year': 2023, 'birth_year': 1990})
    assert obj.default_parameters == {'age': 33,
                                      'birth_year': 1990,
                                      'current_year': 2023,
                                      'height': 180,
                                      'weight': 70}

    assert obj.set_parameters({'age': 29}).default == {'age': 33,
                                                       'birth_year': 1990,
                                                       'current_year': 2023,
                                                       'height': 180,
                                                       'weight': 70}
    assert obj.set_parameters({'age': 29}).user == {}

    assert obj.set_parameters({'current_year': 2024}).default == {'birth_year': 1990, 'height': 180, 'weight': 70}
    assert obj.set_parameters({'current_year': 2024}).user == {'age': 34, 'current_year': 2024}


def test_uses_parameter():
    smith = ParameterizedObject(name='Smith')
    alice = ParameterizedObject(parameters={'age': 27, 'weight': 70}, name='Alice')
    bob = ParameterizedObject(parameters={'age': 25, 'height': 176}, name='Bob')
    smith.add_child(alice)
    smith.add_child(bob)

    assert smith.uses_parameter('Bob' + d + 'age')
    assert smith.uses_parameter('weight')
    assert smith.uses_parameter('age')
    assert smith.uses_parameter('Alice' + d + 'age')
