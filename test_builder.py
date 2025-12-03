from vln_dataset_builder import Builder

builder = Builder()
for key, example in builder._generate_examples(r"D:\Server\OpenFlyData\liujunli_1___OpenFly\raw\Annotation\train.json"):
    print(example)
    break