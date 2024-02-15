from zpgenerator.time.parameters.collection import *
from zpgenerator.time.parameters import Parameters

d = Parameters.DELIMITER

def test_collection_init_setter():
    coll = ParameterizedCollection()
    assert coll._objects == []
    coll.objects = ParameterizedObject(parameters={'age': 25}, name='Bob')
    assert len(coll.objects) == 1
    assert coll._children.children == coll._objects
    assert coll.parameters == ['Bob' + d + 'age']
    assert coll.parameter_tree() == {'Bob': {'age': 25}}


def test_collection_add_objects():
    coll = ParameterizedCollection(name='Smith,')
    coll.add(ParameterizedObject(parameters={'age': 25}, name='Bob'))
    assert len(coll._objects) == 1
    assert len(coll._children.children)== 1
    coll.add(ParameterizedObject(parameters={'age': 27}, name='Alice'))
    assert coll.parameters == ['Alice' + d + 'age', 'Bob' + d + 'age']
    assert coll.parameter_tree({'age': 23}) == {'Alice': {'age': 23}, 'Bob': {'age': 23}}


def test_collection_add_direct():
    coll1 = ParameterizedCollection(name='Smith,')
    coll1.add(ParameterizedObject(parameters={'age': 32}, name='Bob'))
    coll1.add(ParameterizedObject(parameters={'age': 30}, name='Alice'))

    coll2 = ParameterizedCollection(name='Kids,')
    coll2.add(ParameterizedObject(parameters={'age': 8}, name='Charlie'))
    coll2.add(ParameterizedObject(parameters={'age': 3}, name='Eve'))

    coll3 = coll1 + coll2 # merges, and replaces name information of the second collection with the first

    assert coll3.parameters == ['Alice' + d + 'age', 'Bob' + d + 'age', 'Charlie' + d + 'age', 'Eve' + d + 'age']
    assert coll3.parameter_tree() == {'Alice': {'age': 30}, 'Bob': {'age': 32}, 'Charlie': {'age': 8},
                                      'Eve': {'age': 3}}
