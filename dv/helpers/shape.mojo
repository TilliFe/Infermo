
struct Vec:
    var shape: DynamicVector[Int]

    fn __init__(inout self, *_shape: Int):
        let v = VariadicList[Int](_shape)
        let len = len(v)
        self.shape = DynamicVector[Int](0)
        for i in range(len):
            self.shape.push_back(v[i])

    fn get(self) -> DynamicVector[Int]:
        return self.shape

fn shape(*_shape: Int) -> DynamicVector[Int]:
    let v = VariadicList[Int](_shape)
    let len = len(v)
    var shape = DynamicVector[Int](0)
    for i in range(len):
        shape.push_back(v[i])
    return shape