##################################################################################################################
# Compute Graph: Stores all nodes and orchestrates memory allocation efficiently.
##################################################################################################################


# Imports
from memory import memset_zero
from math import log, log2, exp, exp2, ceil, max, abs
from algorithm import vectorize

from .node import Node
from .utils import Vector, get_broadcasted_shape_for_ew_op
from .utils.shape import shape
from .kernels import *

alias VectorF32 = DTypePointer[DType.float32]
alias VectorInt = Vector[Int]
alias DTVector = Vector[VectorF32]
alias NodeVector = Vector[Pointer[Node]]
alias nelts = simdwidthof[DType.float32]()

alias unary_op = fn (b: Node, a: Node) -> None
alias binary_op = fn (c: Node, a: Node, b: Node) -> None
alias view_op = fn (b: Node, a: Node) -> None
alias reduce_op = fn (c: Node, a: Node, b: Node) -> None
alias op_tuple = Tuple[StringRef, unary_op, binary_op, view_op, reduce_op]


fn unary(b: Node, a: Node):
    pass  # elementwise unary


fn binary(c: Node, a: Node, b: Node):
    pass  # elementwise binary


fn view(b: Node, a: Node):
    pass  # change view on Pointer[Node], no computation


fn reduce(c: Node, a: Node, b: Node):
    pass  # reductions, change size of Pointer[Node]


@register_passable("trivial")
struct Graph:
    var nodes: Pointer[NodeVector]
    var memory_pool: Pointer[DTVector]
    var memory_pool_manager: Pointer[VectorInt]
    var free_node_ids: Pointer[VectorInt]
    var free_data_ids: Pointer[VectorInt]
    var last_node_id: Pointer[Int]
    var kernels: Pointer[op_tuple]
    var forward_order: Pointer[VectorInt]
    var grad_nodes_order: Pointer[VectorInt]
    var compiled: Pointer[Bool]

    fn __init__() -> Self:
        let nodes = Pointer[NodeVector].alloc(1)
        nodes.store(NodeVector())

        let memory_pool = Pointer[DTVector].alloc(1)
        memory_pool.store(DTVector())

        let memory_pool_manager = Pointer[VectorInt].alloc(30)
        for i in range(30):
            memory_pool_manager.store(i, VectorInt())

        let free_node_ids = Pointer[VectorInt].alloc(1)
        free_node_ids.store(VectorInt())

        let free_data_ids = Pointer[VectorInt].alloc(1)
        free_data_ids.store(VectorInt())

        let last_node_id = Pointer[Int].alloc(1)
        last_node_id.store(-1)

        let kernels = Pointer[op_tuple].alloc(70)
        # unary operators
        kernels.store(0, op_tuple("cos", fw_cos, binary, view, reduce))
        kernels.store(1, op_tuple("bwcos", bw_cos, binary, view, reduce))
        kernels.store(2, op_tuple("sin", fw_sin, binary, view, reduce))
        kernels.store(3, op_tuple("bwsin", bw_sin, binary, view, reduce))
        kernels.store(4, op_tuple("tan", fw_tan, binary, view, reduce))
        kernels.store(5, op_tuple("bwtan", bw_tan, binary, view, reduce))
        kernels.store(6, op_tuple("acos", fw_acos, binary, view, reduce))
        kernels.store(7, op_tuple("bwacos", bw_acos, binary, view, reduce))
        kernels.store(8, op_tuple("asin", fw_asin, binary, view, reduce))
        kernels.store(9, op_tuple("bwasin", bw_asin, binary, view, reduce))
        kernels.store(10, op_tuple("atan", fw_atan, binary, view, reduce))
        kernels.store(11, op_tuple("bwatan", bw_atan, binary, view, reduce))
        kernels.store(12, op_tuple("cosh", fw_cosh, binary, view, reduce))
        kernels.store(13, op_tuple("bwcosh", bw_cosh, binary, view, reduce))
        kernels.store(14, op_tuple("sinh", fw_sinh, binary, view, reduce))
        kernels.store(15, op_tuple("bwsinh", bw_sinh, binary, view, reduce))
        kernels.store(16, op_tuple("tanh", fw_tanh, binary, view, reduce))
        kernels.store(17, op_tuple("bwtanh", bw_tanh, binary, view, reduce))
        kernels.store(18, op_tuple("log", fw_log, binary, view, reduce))
        kernels.store(19, op_tuple("bwlog", bw_log, binary, view, reduce))
        kernels.store(20, op_tuple("log2", fw_log2, binary, view, reduce))
        kernels.store(21, op_tuple("bwlog", bw_log2, binary, view, reduce))
        kernels.store(22, op_tuple("exp", fw_exp, binary, view, reduce))
        kernels.store(23, op_tuple("bwexp", bw_exp, binary, view, reduce))
        kernels.store(24, op_tuple("exp2", fw_exp2, binary, view, reduce))
        kernels.store(25, op_tuple("bwexp2", bw_exp2, binary, view, reduce))
        kernels.store(26, op_tuple("sqrt", fw_sqrt, binary, view, reduce))
        kernels.store(27, op_tuple("bwsqrt", bw_sqrt, binary, view, reduce))
        kernels.store(28, op_tuple("abs", fw_abs, binary, view, reduce))
        kernels.store(29, op_tuple("bwabs", bw_abs, binary, view, reduce))
        kernels.store(30, op_tuple("relu", fw_relu, binary, view, reduce))
        kernels.store(31, op_tuple("bwrelu", bw_relu, binary, view, reduce))
        kernels.store(32, op_tuple("copy", fw_copy, binary, view, reduce))
        kernels.store(33, op_tuple("bwcopy", bw_copy, binary, view, reduce))

        # binary
        kernels.store(34, op_tuple("add", unary, fw_add, view, reduce))
        kernels.store(35, op_tuple("bwadd", unary, bw_add, view, reduce))
        kernels.store(36, op_tuple("sub", unary, fw_sub, view, reduce))
        kernels.store(37, op_tuple("bwsub", unary, bw_sub, view, reduce))
        kernels.store(38, op_tuple("mul", unary, fw_mul, view, reduce))
        kernels.store(39, op_tuple("bwmul", unary, bw_mul, view, reduce))
        kernels.store(40, op_tuple("div", unary, fw_div, view, reduce))
        kernels.store(41, op_tuple("bwdiv", unary, bw_div, view, reduce))
        kernels.store(42, op_tuple("pow", unary, fw_pow, view, reduce))
        kernels.store(43, op_tuple("bwpow", unary, bw_pow, view, reduce))
        kernels.store(44, op_tuple("mmul", unary, fw_mmul, view, reduce))
        kernels.store(45, op_tuple("bwmmul", unary, bw_mmul, view, reduce))

        # view
        kernels.store(46, op_tuple("reshape", fw_reshape, binary, view, reduce))
        kernels.store(47, op_tuple("bwreshape", bw_reshape, binary, view, reduce))
        kernels.store(48, op_tuple("transpose", fw_transpose, binary, view, reduce))
        kernels.store(49, op_tuple("bwtranspose", bw_transpose, binary, view, reduce))

        # reduce
        kernels.store(50, op_tuple("sum", fw_sum, binary, view, reduce))
        kernels.store(51, op_tuple("bwsum", bw_sum, binary, view, reduce))
        kernels.store(52, op_tuple("mse", unary, fw_mse, view, reduce))
        kernels.store(53, op_tuple("bwmse", unary, bw_mse, view, reduce))
        kernels.store(54, op_tuple("ce", unary, fw_ce, view, reduce))
        kernels.store(55, op_tuple("bwce", unary, bw_ce, view, reduce))
        kernels.store(56, op_tuple("softmax", fw_softmax, binary, view, reduce))
        kernels.store(57, op_tuple("bwsoftmax", bw_softmax, binary, view, reduce))

        kernels.store(58, op_tuple("conv2d", unary, conv_2d, view, reduce))
        kernels.store(59, op_tuple("bwconv2d", unary, bw_conv_2d, view, reduce))
        kernels.store(60, op_tuple("maxpool2dd", max_pool_2d, binary, view, reduce))
        kernels.store(61, op_tuple("bwmaxpool2d", bw_max_pool_2d, binary, view, reduce))

        let forward_order = Pointer[VectorInt].alloc(1)
        forward_order.store(VectorInt())

        let grad_nodes_order = Pointer[VectorInt].alloc(1)
        grad_nodes_order.store(VectorInt())

        let backward_order = Pointer[VectorInt].alloc(1)
        backward_order.store(VectorInt())

        let compiled = Pointer[Bool].alloc(1)
        compiled.store(False)

        return Graph {
            nodes: nodes,
            memory_pool: memory_pool,
            memory_pool_manager: memory_pool_manager,
            free_node_ids: free_node_ids,
            free_data_ids: free_data_ids,
            last_node_id: last_node_id,
            kernels: kernels,
            forward_order: forward_order,
            grad_nodes_order: grad_nodes_order,
            compiled: compiled,
        }

    # Print out methods
    fn print_memory_pool_manager(self) raises:
        for i in range(30):
            let ceiled_cap = exp2(Float32(i)).to_int()
            print_no_newline("    cap:", ceiled_cap)
            print_no_newline(" - data_ids: [")
            for j in range(self.memory_pool_manager.load(i).len.load()):
                print_no_newline(self.memory_pool_manager.load(i).load(j))
                if j < self.memory_pool_manager.load(i).len.load() - 1:
                    print_no_newline(", ")
            print("]")

    fn print(self) raises:
        print("\nGraph (Nodes):")
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node == Pointer[Node].get_null():
                continue
            node.load().print()

    # Helper methods...
    # Select a free node_id and data_id: This prevents memory fragmentation over time
    fn get_free_node_id(self) raises -> Int:
        var fid: Int = 0
        if self.free_node_ids.load().len.load() > 0:
            fid = self.free_node_ids.load().pop_back()
        else:
            fid = self.nodes.load().len.load()
        return fid

    fn get_free_node_id_no_pop(self) raises -> Int:
        var fid: Int = 0
        if self.free_node_ids.load().len.load() > 0:
            fid = self.free_node_ids.load().load(
                self.free_node_ids.load().len.load() - 1
            )
        else:
            fid = self.nodes.load().len.load()
        return fid

    fn get_free_data_id(self) raises -> Int:
        if self.free_data_ids.load().len.load() > 0:
            return self.free_data_ids.load().pop_back()
        return self.memory_pool.load().len.load()

    fn load_ceiled_cap(self, cap: Int) raises -> Int:
        return exp2(ceil(log2(Float32(cap)))).to_int()

    fn get_index(self, cap: Int) raises -> Int:
        return ceil(log2(Float32(cap))).to_int()

    # Create a new node in the compute graph.
    fn node(
        self,
        shape: Vector[Int],
        is_static: Bool,
        is_single: Bool,
        checkpoint: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parent_ptrs: Pointer[Node],
    ) raises -> Pointer[Node]:
        let node = Node(self.get_free_node_id(), shape, is_static)
        node.operator_id_ptr.store(operator_id)
        node.checkpoint_ptr.store(checkpoint)
        node.is_single_ptr.store(is_single)
        node.grad_operator_id_ptr.store(operator_id + 1)
        node.other_params_ptr.store(other_params.copy())
        let node_ptr = Pointer[Node].alloc(1)
        node_ptr.store(node)

        # store parents and children
        for i in range(len(parent_ptrs)):
            node.add_parent(parent_ptrs[i].load().load_id())
            parent_ptrs[i].load().add_child(node.load_id())
            parent_ptrs[i].load().incr_dependencies()

        self.get_free_data_ptr(node_ptr)

        for i in range(len(parent_ptrs)):
            if parent_ptrs[i].load().dependencies_ptr.load() == 0:
                _ = self.forward_recursive(parent_ptrs[i])

        # Add node to graph
        let node_id = node_ptr.load().load_id()
        if node_id < self.nodes.load().len.load():
            self.nodes.load().store(node_id, node_ptr)
        else:
            self.nodes.load().push_back(node_ptr)

        return node_ptr

    fn node(
        self,
        shape: DynamicVector[Int],
        is_static: Bool,
        is_single: Bool,
        checkpoint: Bool,
        operator_id: Int,
        other_params: Vector[Int],
        *parent_ptrs: Pointer[Node],
    ) raises -> Pointer[Node]:
        let _shape = Vector[Int]()
        for i in range(len(shape)):
            _shape.push_back(shape[i])
        let node = Node(self.get_free_node_id(), _shape, is_static)
        node.checkpoint_ptr.store(checkpoint)
        node.is_single_ptr.store(is_single)
        node.operator_id_ptr.store(operator_id)
        node.grad_operator_id_ptr.store(operator_id + 1)
        node.other_params_ptr.store(other_params.copy())
        let node_ptr = Pointer[Node].alloc(1)
        node_ptr.store(node)

        # storeup parents and children
        for i in range(len(parent_ptrs)):
            node.add_parent(parent_ptrs[i].load().load_id())
            parent_ptrs[i].load().add_child(node.load_id())
            parent_ptrs[i].load().incr_dependencies()

        self.get_free_data_ptr(node_ptr)

        for i in range(len(parent_ptrs)):
            if parent_ptrs[i].load().dependencies_ptr.load() == 0:
                _ = self.forward_recursive(parent_ptrs[i])

        # Add node to graph
        let node_id = node_ptr.load().load_id()
        if node_id < self.nodes.load().len.load():
            self.nodes.load().store(node_id, node_ptr)
        else:
            self.nodes.load().push_back(node_ptr)

        return node_ptr

    # Given a node in the graph, select some allocated memory for it, if possible point to the same
    # data as the parent does, otherwise try to select from the memory_pool of freed up data vector,
    # and only if both fail, allocate a new vector of Floating Point numbers for the data.
    fn get_free_data_ptr(self, node: Pointer[Node], unique: Bool = False) raises:
        if node.load().data_id.load() != -1:
            return

        var idx = -1
        # if not node.load().is_static_ptr.load() and not node.load().checkpoint_ptr.load():
        # try to load data from one of the parents first
        for i in range(node.load().parents_ptr.load().len.load()):
            let ind = node.load().parents_ptr.load().load(i)
            let parent = self.nodes.load().load(node.load().load_parent_id(i))
            if (
                self.load_ceiled_cap(parent.load().cap_ptr.load())
                == self.load_ceiled_cap(node.load().cap_ptr.load())
                and parent.load().dependencies_ptr.load() == 1
                and not parent.load().is_static_ptr.load()
                and not node.load().is_static_ptr.load()
                and not parent.load().checkpoint_ptr.load()
                and not node.load().checkpoint_ptr.load()
                and not unique
                and not parent.load().is_single_ptr.load()
                and not node.load().is_single_ptr.load()
            ):
                # print(node.load().load_id(),"got data from parent", parent.load().load_id(),"- data_id:", parent.load().data_id.load())
                node.load().data_id.store(parent.load().data_id.load())
                node.load().data.store(
                    0, self.memory_pool.load().load(node.load().data_id.load())
                )
                idx = i
                break

        for i in range(node.load().parents_ptr.load().len.load()):
            if i == idx:
                continue
            else:
                let parent = self.nodes.load().load(node.load().load_parent_id(i))
                parent.load().decr_dependencies()

        # if data not shared by parent, load data from memory_pool or create new
        if idx == -1:
            let index = self.get_index(node.load().cap_ptr.load())
            if self.memory_pool_manager.load(index).len.load() > 0:
                let data_id = self.memory_pool_manager.load(index).pop_back()
                node.load().data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load())

                # reset all entries to zero
                node.load().data.store(
                    0, self.memory_pool.load().load(node.load().data_id.load())
                )
                memset_zero(node.load().data.load(0), ceiled_cap)
                # print(node.load().load_id(),"got data from storage", data_id)
            else:
                let data_id = self.get_free_data_id()
                node.load().data_id.store(data_id)
                let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load() + 1)
                let new_data_ptr = VectorF32.alloc(ceiled_cap)
                if data_id == self.memory_pool.load().len.load():
                    self.memory_pool.load().push_back(new_data_ptr)
                else:
                    self.memory_pool.load().data.load().store(data_id, new_data_ptr)

                # reset all entries to zero
                node.load().data.store(
                    0, self.memory_pool.load().load(node.load().data_id.load())
                )
                memset_zero(node.load().data.load(0), ceiled_cap)
                # print(node.load().load_id(),"got new data", data_id)

    fn get_free_grad_ptr(self, node: Pointer[Node]) raises:
        if node.load().grad_id.load() != -1:
            return

        let index = self.get_index(node.load().cap_ptr.load())
        if self.memory_pool_manager.load(index).len.load() > 0:
            let grad_id = self.memory_pool_manager.load(index).pop_back()
            node.load().grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load())

            # reset all entries to zero
            node.load().data.store(
                1, self.memory_pool.load().load(node.load().grad_id.load())
            )
            memset_zero(node.load().data.load(1), ceiled_cap)
            # print(node.load().load_id(),"got grad from storage", grad_id)
        else:
            let grad_id = self.get_free_data_id()
            node.load().grad_id.store(grad_id)
            let ceiled_cap = self.load_ceiled_cap(node.load().cap_ptr.load())
            let new_grad_ptr = VectorF32.alloc(ceiled_cap)
            if grad_id == self.memory_pool.load().len.load():
                self.memory_pool.load().push_back(new_grad_ptr)
            else:
                self.memory_pool.load().data.load().store(grad_id, new_grad_ptr)

            # reset all entries to zero
            node.load().data.store(
                1, self.memory_pool.load().load(node.load().grad_id.load())
            )
            memset_zero(node.load().data.load(1), ceiled_cap)
            # print(node.load().load_id(),"got new grad", grad_id)

    # If a node is being released, we can reuse its data and its ids.
    fn release_data(self, node_ptr: Pointer[Node]) raises:
        if (
            node_ptr.load().is_static_ptr.load()
            or node_ptr.load().checkpoint_ptr.load()
            or node_ptr.load().is_single_ptr.load()
            or node_ptr.load().data_id.load() == -1
        ):
            return

        if node_ptr.load().dependencies_ptr.load() == 0:
            let index = self.get_index(node_ptr.load().cap_ptr.load())
            let data_id = node_ptr.load().data_id.load()
            self.memory_pool_manager.load(index).push_back(data_id)
            node_ptr.load().data_id.store(-1)
            node_ptr.load().dependencies_ptr.store(
                node_ptr.load().children_ptr.load().len.load()
            )
            node_ptr.load().computed_ptr.store(False)
            # print("    -> release data", data_id, "of node", node_ptr.load().load_id())

    fn release_data_forced(self, node_ptr: Pointer[Node]) raises:
        if node_ptr.load().is_static_ptr.load() or node_ptr.load().data_id.load() == -1:
            return
        let index = self.get_index(node_ptr.load().cap_ptr.load())
        let data_id = node_ptr.load().data_id.load()
        self.memory_pool_manager.load(index).push_back(data_id)
        node_ptr.load().data_id.store(-1)
        node_ptr.load().computed_ptr.store(False)
        node_ptr.load().dependencies_ptr.store(
            node_ptr.load().children_ptr.load().len.load()
        )
        # print("    -> release data (forced)", data_id, "of node", node_ptr.load().load_id())

    fn release_grad_forced(self, node_ptr: Pointer[Node]) raises:
        if node_ptr.load().is_static_ptr.load() or node_ptr.load().grad_id.load() == -1:
            return
        let index = self.get_index(node_ptr.load().cap_ptr.load())
        let grad_id = node_ptr.load().grad_id.load()
        self.memory_pool_manager.load(index).push_back(grad_id)
        node_ptr.load().grad_id.store(-1)
        node_ptr.load().grad_computed_ptr.store(False)
        # print("    -> release grad (forced)", grad_id, "of node", node_ptr.load().load_id())

    # This methods clears up al intermediate nodes, which where stored during the forward pass.
    # This methods shall only be called at  very end or beginning of a compute pass
    fn clear_cache(self, reset_static_nodes: Bool = False) raises:
        if self.last_node_id.load() != -1:
            let node_ptr = self.nodes.load().load(self.last_node_id.load())
            self.release_data_forced(node_ptr)

        # remove duplicate nodes
        for i in range(self.nodes.load().len.load() - 1):
            if self.nodes.load().load(i) == Pointer[Node].get_null():
                continue
            for j in range(i + 1, self.nodes.load().len.load()):
                if (
                    self.nodes.load().load(i).load().load_id()
                    == self.nodes.load().load(j).load().load_id()
                ):
                    self.nodes.load().store(i, Pointer[Node].get_null())
                    break

        # remove duplicate memory pointers
        for i in range(self.memory_pool.load().len.load()):
            let array = self.memory_pool.load().load(i)
            for j in range(i + 1, self.memory_pool.load().len.load()):
                let other = self.memory_pool.load().load(j)
                if array == other:
                    self.memory_pool.load().store(i, VectorF32.get_null())

        # free up unused data and grad data from memory pool
        let deletable_data = Vector[Bool](self.memory_pool.load().len.load())
        for i in range(self.memory_pool.load().len.load()):
            deletable_data.store(i, True)
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node == Pointer[Node].get_null():
                continue

            if node.load().is_static_ptr.load():
                if node.load().data_id.load() != -1:
                    deletable_data.store(node.load().data_id.load(), False)
                if node.load().grad_id.load() != -1:
                    deletable_data.store(node.load().grad_id.load(), False)

        for i in range(deletable_data.len.load()):
            if (
                deletable_data.load(i)
                and not self.memory_pool.load().load(i) == VectorF32.get_null()
            ):
                self.memory_pool.load().load(i).free()
        deletable_data.free()

        # var count_back = self.nodes.load().len.load()-1
        for i in range(self.nodes.load().len.load() - 1, -1, -1):
            let node_ptr = self.nodes.load().load(i)
            if node_ptr == Pointer[Node].get_null():
                continue

            # remove entire node if mode is not static
            if not node_ptr.load().load_is_static():
                self.free_node_ids.load().push_back(node_ptr.load().load_id())

                # free all pointers deeply
                node_ptr.load().id_ptr.free()
                node_ptr.load().data_id.free()
                node_ptr.load().grad_id.free()
                node_ptr.load().data.free()
                node_ptr.load().parents_ptr.load().free()
                node_ptr.load().parents_ptr.free()
                node_ptr.load().children_ptr.load().free()
                node_ptr.load().children_ptr.free()
                node_ptr.load().dependencies_ptr.free()
                node_ptr.load().is_static_ptr.free()
                node_ptr.load().computed_ptr.free()
                node_ptr.load().grad_computed_ptr.free()
                node_ptr.load().operator_id_ptr.free()
                node_ptr.load().grad_operator_id_ptr.free()
                node_ptr.load().requires_grad_ptr.free()
                node_ptr.load().tmp_visited_ptr.free()
                node_ptr.load().checkpoint_ptr.free()

                node_ptr.load().cap_ptr.free()
                node_ptr.load().num_dims_ptr.free()
                node_ptr.load().shape_ptr.load().free()
                node_ptr.load().shape_ptr.free()
                node_ptr.load().strides_ptr.load().free()
                node_ptr.load().strides_ptr.free()
                node_ptr.load().other_params_ptr.load().free()
                node_ptr.load().other_params_ptr.free()

                node_ptr.free()
            else:
                node_ptr.load().children_ptr.load().clear()
                node_ptr.load().parents_ptr.load().clear()
                node_ptr.load().dependencies_ptr.store(0)
                node_ptr.load().id_ptr.store(0)
                node_ptr.load().data_id.store(0)
                node_ptr.load().grad_id.store(0)

    # clear the graph carefully, only dynamic nodes are freed deeply, static nodes are kept alive.
    fn clear(self, reset_static_nodes: Bool = False) raises:
        self.clear_cache(reset_static_nodes)

        # free all graph related parts
        self.nodes.load().free()
        self.nodes.free()
        self.memory_pool.load().free()
        self.memory_pool.free()
        for i in range(30):
            self.memory_pool_manager.load(i).free()
        self.memory_pool_manager.free()
        self.free_node_ids.load().free()
        self.free_node_ids.free()
        self.free_data_ids.load().free()
        self.free_data_ids.free()
        self.last_node_id.free()
        self.kernels.free()
        self.forward_order.load().free()
        self.forward_order.free()
        self.compiled.free()

    # depth first search starting at the current node
    fn forward_recursive(
        self, node_ptr: Pointer[Node], keep_forward_order: Bool = False
    ) raises -> Pointer[Node]:
        # base case
        if (
            node_ptr.load().load_computed()
        ):  # or node_ptr.load().load_num_parents() == 0):
            return node_ptr

        # print(node_ptr.load().load_id())

        # go into depth
        let operator_id = node_ptr.load().operator_id_ptr.load()
        if node_ptr.load().load_num_parents() == 1:
            let parent1_ptr = self.forward_recursive(
                self.nodes.load().load(node_ptr.load().load_parent_id(0)),
                keep_forward_order,
            )
            self.get_free_data_ptr(node_ptr)
            self.kernels.load(operator_id).get[1, unary_op]()(
                node_ptr.load(), parent1_ptr.load()
            )
            self.release_data(parent1_ptr)
        else:
            let parent1_ptr = self.forward_recursive(
                self.nodes.load().load(node_ptr.load().load_parent_id(0)),
                keep_forward_order,
            )
            let parent2_ptr = self.forward_recursive(
                self.nodes.load().load(node_ptr.load().load_parent_id(1)),
                keep_forward_order,
            )
            self.get_free_data_ptr(node_ptr)
            self.kernels.load(operator_id).get[2, binary_op]()(
                node_ptr.load(), parent1_ptr.load(), parent2_ptr.load()
            )

            self.release_data(parent1_ptr)
            self.release_data(parent2_ptr)

        # store forward order in case we do a static pass later
        if keep_forward_order:
            self.forward_order.load().push_back(node_ptr.load().load_id())

        # mark node as computed
        node_ptr.load().computed_ptr.store(True)

        return node_ptr

    # Computes a forward pass based on the already computed nodes
    fn forward(
        self, node_ptr: Pointer[Node], keep_forward_order: Bool = False
    ) raises -> Pointer[Node]:
        self.last_node_id.store(node_ptr.load().load_id())
        self.compiled.store(False)
        let res = self.forward_recursive(node_ptr, keep_forward_order)
        return res

    ### backward stuff ##################################################################################

    # Computes a full forward pass on a static graph based ont the forward tape generated by one dynamic pass.
    fn forward_static(self, node_ptr: Pointer[Node]) raises -> Pointer[Node]:
        self.release_data_forced(node_ptr)

        # clean graph
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            if node.load().is_single_ptr.load():
                continue

            if not node.load().is_static_ptr.load():
                node.load().computed_ptr.store(False)
                node.load().grad_id.store(-1)
                node.load().data_id.store(-1)
            node.load().dependencies_ptr.store(
                node.load().children_ptr.load().len.load()
            )

        _ = self.forward_recursive(node_ptr)

        return self.nodes.load().load(self.last_node_id.load())

    # depth first search starting at the current node with no relaese of any intermediate memory -> used for gradient checkpointing
    fn forward_recursive_graph_slice(
        self, node_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        # print("\ntry node",node_ptr.load().load_id())
        # base case
        if (
            node_ptr.load().computed_ptr.load()
        ):  # or node_ptr.load().is_single_ptr.load():# or node_ptr.load().checkpoint_ptr.load() or node_ptr.load().is_static_ptr.load(): # or node_ptr.load().load_num_parents() == 0:
            return node_ptr
        # print("compute", node_ptr.load().load_id())

        # go into depth
        let operator_id = node_ptr.load().operator_id_ptr.load()
        if node_ptr.load().load_num_parents() == 1:
            let parent1_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node_ptr.load().parents_ptr.load().load(0))
            )
            self.get_free_data_ptr(node_ptr, True)
            # print("   -> compute slice node",node_ptr.load().load_id(),"from", parent1_ptr.load().load_id())

            self.kernels.load(operator_id).get[1, unary_op]()(
                node_ptr.load(), parent1_ptr.load()
            )
        else:
            let parent1_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node_ptr.load().parents_ptr.load().load(0))
            )
            let parent2_ptr = self.forward_recursive_graph_slice(
                self.nodes.load().load(node_ptr.load().parents_ptr.load().load(1))
            )
            # print("   -> compute slice node",node_ptr.load().load_id(),"from", parent1_ptr.load().load_id(),"and",parent2_ptr.load().load_id())

            self.get_free_data_ptr(node_ptr, True)
            self.kernels.load(operator_id).get[2, binary_op]()(
                node_ptr.load(), parent1_ptr.load(), parent2_ptr.load()
            )

        # mark node as computed
        node_ptr.load().computed_ptr.store(True)

        return node_ptr

    fn backward_recursive(self, node_ptr: Pointer[Node]) raises -> Pointer[Node]:
        # base case
        if node_ptr.load().grad_computed_ptr.load():
            return node_ptr

        for i in range(node_ptr.load().children_ptr.load().len.load()):
            let child_id = node_ptr.load().children_ptr.load().load(i)
            let child_ptr = self.nodes.load().load(child_id)
            _ = self.backward_recursive(child_ptr)

            let grad_operator_id = child_ptr.load().grad_operator_id_ptr.load()
            if child_ptr.load().parents_ptr.load().len.load() == 1:
                let parent1_ptr = self.nodes.load().load(
                    child_ptr.load().load_parent_id(0)
                )
                _ = self.forward_recursive_graph_slice(parent1_ptr)

                if parent1_ptr.load().grad_id.load() == -1:
                    self.get_free_grad_ptr(parent1_ptr)

                parent1_ptr.load().grad_computed_ptr.store(True)

                self.kernels.load(grad_operator_id).get[1, unary_op]()(
                    child_ptr.load(), parent1_ptr.load()
                )

            else:
                let parent1_ptr = self.nodes.load().load(
                    child_ptr.load().load_parent_id(0)
                )
                let parent2_ptr = self.nodes.load().load(
                    child_ptr.load().load_parent_id(1)
                )

                _ = self.forward_recursive_graph_slice(parent1_ptr)
                _ = self.forward_recursive_graph_slice(parent2_ptr)
                # print(node_ptr.load().load_id(),"            -> compute grads of",parent1_ptr.load().load_id(),"and",parent2_ptr.load().load_id())

                if parent1_ptr.load().grad_id.load() == -1:
                    self.get_free_grad_ptr(parent1_ptr)
                if parent2_ptr.load().grad_id.load() == -1:
                    self.get_free_grad_ptr(parent2_ptr)

                parent1_ptr.load().grad_computed_ptr.store(True)
                parent2_ptr.load().grad_computed_ptr.store(True)

                self.kernels.load(grad_operator_id).get[2, binary_op]()(
                    child_ptr.load(), parent1_ptr.load(), parent2_ptr.load()
                )

            if child_ptr.load().load_id() != self.last_node_id.load():
                self.release_data_forced(child_ptr)
            self.release_grad_forced(child_ptr)

        return node_ptr

    # bfs starting at the last node, to find order from where to start dfs style grad computation
    fn find_grad_nodes_order(self, node_ptr: Pointer[Node]) raises:
        self.grad_nodes_order.store(Vector[Int]())
        for i in range(self.nodes.load().len.load()):
            self.nodes.load().load(i).load().tmp_visited_ptr.store(False)
        self.grad_nodes_order.load().clear()

        var backward = DynamicVector[Int]()
        backward.push_back(node_ptr.load().load_id())
        var it = 0
        while it < len(backward):
            let currId = backward[it]
            let curr = self.nodes.load().load(currId)
            for i in range(curr.load().parents_ptr.load().len.load()):
                let parId = curr.load().parents_ptr.load().load(i)
                let par = self.nodes.load().load(parId)
                if not par.load().tmp_visited_ptr.load():
                    backward.push_back(parId)
            if (
                curr.load().requires_grad_ptr.load()
                or curr.load().checkpoint_ptr.load()
            ):  # and not curr.load().is_single_ptr.load():
                self.grad_nodes_order.load().push_back(currId)
            self.nodes.load().load(currId).load().tmp_visited_ptr.store(True)
            it += 1

    fn backward(self, node_ptr: Pointer[Node]) raises:
        # set backward order
        self.find_grad_nodes_order(node_ptr)

        self.last_node_id.store(node_ptr.load().load_id())

        # clean graph
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i)
            node.load().grad_computed_ptr.store(False)

            if (
                node.load().is_single_ptr.load()
                or node.load().load_id() == self.last_node_id.load()
            ):
                continue

            if not node.load().is_static_ptr.load():
                node.load().grad_id.store(-1)
                if not node.load().checkpoint_ptr.load():
                    node.load().computed_ptr.store(False)
                    node.load().data_id.store(-1)
            else:
                if node.load().grad_id.load() != -1:
                    memset_zero(
                        node.load().data.load(1),
                        self.load_ceiled_cap(node.load().cap_ptr.load()),
                    )

        # perform recursive backtracking
        self.get_free_grad_ptr(node_ptr)
        node_ptr.load().fill_grad(1.0)
        node_ptr.load().grad_computed_ptr.store(True)
        for i in range(self.grad_nodes_order.load().len.load()):
            let curr_node_ptr = self.nodes.load().load(
                self.grad_nodes_order.load().load(i)
            )
            # print("starting grad computation from",curr_node_ptr.load().load_id())
            _ = self.backward_recursive(curr_node_ptr)

    # optimizer
    fn optimizer_step(self, learning_rate: Float32, type: String) raises:
        for i in range(self.nodes.load().len.load()):
            let node = self.nodes.load().load(i).load()
            if (
                type == "sgd"
                and node.requires_grad_ptr.load()
                and node.grad_computed_ptr.load()
            ):

                @parameter
                fn v_sgd_update[_nelts: Int](i: Int):
                    node.store_data[_nelts](
                        i,
                        node.load_data[_nelts](i)
                        - node.load_grad[_nelts](i) * learning_rate,
                    )

                vectorize[nelts, v_sgd_update](node.load_cap())

    ####################################################################################
    # unary operators
    ####################################################################################
    fn cos(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 0
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sin(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 2
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn tan(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 4
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn acos(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 6
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn asin(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 8
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn atan(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 10
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn cosh(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 12
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sinh(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 14
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn tanh(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 16
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn log(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 18
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn log2(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 20
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn exp(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 22
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn exp2(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 24
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn sqrt(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 26
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn abs(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 28
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn relu(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 30
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn copy(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 32
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, True, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    ####################################################################################
    # binary operators
    ####################################################################################
    fn add(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 34
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn sub(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 36
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn mul(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 38
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn div(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 40
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn pow(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 42
        let checkpoint = False
        let shape = get_broadcasted_shape_for_ew_op(a, b)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn mmul(self, a: Pointer[Node], b: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 44
        let checkpoint = True
        var shape = get_broadcasted_shape_for_ew_op(a, b)
        shape[len(shape) - 2] = (
            a.load().shape_ptr.load().copy().load(a.load().num_dims_ptr.load() - 2)
        )
        shape[len(shape) - 1] = (
            b.load().shape_ptr.load().copy().load(b.load().num_dims_ptr.load() - 1)
        )
        # raise if shapes don't fit
        if a.load().shape_ptr.load().load(
            a.load().num_dims_ptr.load() - 1
        ) != b.load().shape_ptr.load().load(b.load().num_dims_ptr.load() - 2):
            raise "Shapes don't fit for matrix multiplication"
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

    fn conv_2d(
        self, a: Pointer[Node], b: Pointer[Node], padding: Int, stride: Int
    ) raises -> Pointer[Node]:  # a: input, b: kernels
        # assumption: a (batch of input images) is of shape (batch_size, channels, width, height)
        #             b (set of kernels) is of shape (num_filters, channels, a, b)

        let a_num_dims = a.load().num_dims_ptr.load()
        let b_num_dims = b.load().num_dims_ptr.load()

        let batch_size = a.load().shape_ptr.load().load(0)
        let in_channels = a.load().shape_ptr.load().load(1)
        let width = a.load().shape_ptr.load().load(2)
        let height = a.load().shape_ptr.load().load(3)

        let out_channels = b.load().shape_ptr.load().load(0)
        if in_channels != b.load().shape_ptr.load().load(1):
            raise "Error (at conv_2d): number of channels must be equal in the input and the kernels"
        let kernel_width = b.load().shape_ptr.load().load(2)
        let kernel_height = b.load().shape_ptr.load().load(3)

        # init result Pointer[Node]
        let shape = shape(
            batch_size,
            out_channels,
            (width - kernel_width + 2 * padding) // stride + 1,
            (height - kernel_height + 2 * padding) // stride + 1,
        )
        let operator_id = 58
        let checkpoint = True
        let other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)
        var c = self.node(
            shape, False, False, checkpoint, operator_id, other_params, a, b
        )

        return c

    fn max_pool_2d(
        self,
        a: Pointer[Node],
        kernel_width: Int,
        kernel_height: Int,
        stride: Int,
        padding: Int,
    ) raises -> Pointer[Node]:
        let new_shape = shape(
            a.load().shape_ptr.load().load(0),
            a.load().shape_ptr.load().load(1),
            (a.load().shape_ptr.load().load(2) - kernel_width + 2 * padding) // stride
            + 1,
            (a.load().shape_ptr.load().load(3) - kernel_height + 2 * padding) // stride
            + 1,
        )
        let operator_id = 60
        let checkpoint = False
        let other_params = Vector[Int]()
        other_params.push_back(padding)
        other_params.push_back(stride)
        other_params.push_back(kernel_width)
        other_params.push_back(kernel_height)

        var b = self.node(
            new_shape, False, False, checkpoint, operator_id, other_params, a
        )

        return b

    ####################################################################################
    # view operators
    ####################################################################################
    fn reshape(
        self, parent1_ptr: Pointer[Node], shape: Vector[Int]
    ) raises -> Pointer[Node]:
        let operator_id = 46
        let checkpoint = False
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn transpose(self, parent1_ptr: Pointer[Node]) raises -> Pointer[Node]:
        let operator_id = 48
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        shape.store(
            shape.len.load() - 2,
            parent1_ptr.load().shape_ptr.load().load(shape.len.load() - 1),
        )
        shape.store(
            shape.len.load() - 1,
            parent1_ptr.load().shape_ptr.load().load(shape.len.load() - 2),
        )
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    ####################################################################################
    # reduction operators
    ####################################################################################
    fn sum(self, parent1_ptr: Pointer[Node], axis: Int) raises -> Pointer[Node]:
        let operator_id = 50
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )

    fn mse(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = 52
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        return self.node(
            shape,
            False,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )

    fn ce(
        self, parent1_ptr: Pointer[Node], parent2_ptr: Pointer[Node]
    ) raises -> Pointer[Node]:
        let operator_id = 54
        let checkpoint = False
        let shape = shape(1)
        let other_params = Vector[Int]()
        return self.node(
            shape,
            False,
            False,
            checkpoint,
            operator_id,
            other_params,
            parent1_ptr,
            parent2_ptr,
        )

    fn softmax(self, parent1_ptr: Pointer[Node], axis: Int) raises -> Pointer[Node]:
        let operator_id = 56
        let checkpoint = False
        let shape = parent1_ptr.load().shape_ptr.load().copy()
        let other_params = Vector[Int]()
        return self.node(
            shape, False, False, checkpoint, operator_id, other_params, parent1_ptr
        )
