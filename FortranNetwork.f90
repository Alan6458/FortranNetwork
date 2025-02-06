module FortranNetwork
implicit none

type net_layer
    ! change from CrapPyNetwork: input layer no longer has biases
    ! Note: this feature may have not been in CrapPyNetwork, I need to check
    real, allocatable, dimension(:,:) :: weights
    real, allocatable, dimension(:) :: biases
    integer :: layer_size
    character :: activation
end type net_layer

type net_out_layer
    real, allocatable, dimension(:) :: out_layer
end type net_out_layer

contains

subroutine init_net(out_net, layer_nodes, activation)
    implicit none

    integer, dimension(:), intent(IN) :: layer_nodes
    type(net_layer), dimension(size(layer_nodes)-1), intent(OUT) :: out_net
    character (len=size(layer_nodes)-1), intent(IN) :: activation
    integer i

    ! Network only needs layers-1 amount of weights/layers
    ! allocating space for each layer of weights and biases
    do i = 1, size(layer_nodes)-1
        ! Again, only needs layers-1 amout of weights/layers which is why it is i and i+1 being used
        ! (I predict this may be an indexing nightmare later on)
        allocate(out_net(i)%weights(layer_nodes(i), layer_nodes(i+1)))
        allocate(out_net(i)%biases(layer_nodes(i+1)))

        ! Assigns values to activation function and layer size
        out_net(i)%activation = activation(i:i)
        out_net(i)%layer_size = layer_nodes(i+1)

        ! generates random floats between 0 and 1 and makes them between -1 and 1
        call random_number(out_net(i)%weights)
        out_net(i)%weights = out_net(i)%weights * 2 - 1

        call random_number(out_net(i)%biases)
        out_net(i)%biases = out_net(i)%biases * 2 - 1
    end do
end subroutine init_net

function fwd_prop(network, input)
    implicit none

    type(net_layer), dimension(:), intent(IN) :: network
    real, dimension(:), intent(IN) :: input
    type(net_out_layer), dimension(size(network)+1) :: fwd_prop
    integer i

    ! The input layer - Should this be included in output?
    ! Again, this may be an indexing issue later on
    allocate(fwd_prop(1)%out_layer(size(input)))
    fwd_prop(1)%out_layer = input
    ! Allocates space for next layer and performs matrix multiplication and adds biases, then does activation
    do i = 2, size(network)+1
        ! Here's (just the beginning of) the indexing nightmare I predicted earlier
        allocate(fwd_prop(i)%out_layer(network(i-1)%layer_size))
        fwd_prop(i)%out_layer = matmul(fwd_prop(i-1)%out_layer, network(i-1)%weights) + network(i-1)%biases
        deallocate(fwd_prop(i-1)%out_layer)
        call activation_func(fwd_prop(i)%out_layer, network(i-1)%activation)
    end do
end function fwd_prop

! This outputs the index of the max output node of an net_out_layer object
function fwd_prop_res(fwd_prop_out)
    implicit none
    
    type(net_out_layer), dimension(:), intent(IN) :: fwd_prop_out
    integer, dimension(1) :: pre_res
    integer fwd_prop_res
    pre_res = maxloc(fwd_prop_out(size(fwd_prop_out))%out_layer)
    fwd_prop_res = pre_res(1)
end function fwd_prop_res

! Note: this is just temporary, may need a faster method
! Need to have function as a variable in the layer type instead of char, like lambda in Python
subroutine activation_func(node_vals, activation_type)
    implicit none

    character (len = 1), intent(IN) :: activation_type
    real, dimension(:), intent(INOUT) :: node_vals
    
    if (activation_type == "s") then
        ! Sigmoid
        node_vals = 1/(1+exp(-node_vals))
    else if (activation_type == "t") then
        ! TanH
        node_vals = tanh(node_vals)
    else if (activation_type == "r") then
        ! ReLU
        ! node_vals = (node_vals + abs(node_vals))/2
        where (node_vals < 0.0) node_vals = 0.0
    else if (activation_type == "S") then
        ! SiLU
        node_vals = node_vals/(1+exp(-node_vals))
    else if (activation_type == "m") then
        ! Softmax
        node_vals = exp(node_vals)/sum(exp(node_vals))
    end if
end subroutine activation_func

subroutine activation_func_derivative(node_vals, activation_type)
    implicit none

    character (len = 1), intent(IN) :: activation_type
    real, dimension(:), intent(INOUT) :: node_vals
    
    ! This is like activation function but the derivative (for backprop)

    if (activation_type == "s") then
        ! Sigmoid
        node_vals = exp(-node_vals)/((1+exp(-node_vals))**2)
    else if (activation_type == "t") then
        ! TanH
        node_vals = 1-(tanh(node_vals)**2)
    else if (activation_type == "r") then
        ! ReLU
        ! node_vals = (node_vals/abs(node_vals)+1)/2
        where (node_vals >= 0.0) node_vals = 1.0
        where (node_vals < 0.0) node_vals = 0.0
    else if (activation_type == "S") then
        ! SiLU
        node_vals = (1+exp(-node_vals)*(1+node_vals))/((1+exp(-node_vals))**2)
    else if (activation_type == "m") then
        ! Softmax (this is gonna be slow but whatever)
        node_vals = (exp(node_vals)/sum(exp(node_vals)))*(1-(exp(node_vals)/sum(exp(node_vals))))
    end if
end subroutine activation_func_derivative

subroutine cost_func(node_vals, goal, cost_type)
    implicit none

    character (len = 1), intent(IN) :: cost_type
    real, dimension(:), intent(IN) :: goal
    real, dimension(:), intent(INOUT) :: node_vals

    if (cost_type == "s") then
        ! Mean Squared Error (MSE)
        node_vals = (node_vals-goal)**2
    else if (cost_type == "a") then
        ! Mean Absolute Error (MAE)
        node_vals = abs(node_vals-goal)
    else if (cost_type == "c") then
        ! Softmax Cross-Entropy Loss
        node_vals = -goal*log(exp(node_vals)/sum(exp(node_vals)))
    end if
end subroutine cost_func

subroutine cost_func_derivative(node_vals, goal, cost_type)
    implicit none

    character (len = 1), intent(IN) :: cost_type
    real, dimension(:), intent(IN) :: goal
    real, dimension(:), intent(INOUT) :: node_vals

    if (cost_type == "s") then
        ! Mean Squared Error (MSE)
        node_vals = 2 * (node_vals-goal)
    else if (cost_type == "a") then
        ! Mean Absolute Error (MAE)
        ! node_vals = node_vals/abs(node_vals)
        where (node_vals >= 0.0) node_vals = 1.0
        where (node_vals < 0.0) node_vals = -1.0
    else if (cost_type == "c") then
        ! Softmax Cross-Entropy Loss
        node_vals = exp(node_vals)/sum(exp(node_vals))-goal
    end if
end subroutine cost_func_derivative

subroutine sgd(sgd_out, net, input, goal, net_cost_func)
    implicit none

    type(net_layer), dimension(:), intent(IN) :: net
    type(net_out_layer), dimension(size(net)+1) :: activations, weighted_sums
    real, dimension(:), intent(IN) :: input
    real, dimension(:), intent(IN) :: goal
    character (len = 1), intent(IN) :: net_cost_func
    type(net_layer), dimension(:), intent(INOUT) :: sgd_out
    integer :: i, j

    ! Forward Propagation (a lot of this code was stolen from fwd_prop)

    ! Allocates space and assignes values to first layer of weighted sums and activations
    allocate(activations(1)%out_layer(size(input)))
    allocate(weighted_sums(1)%out_layer(size(input)))
    activations(1)%out_layer = input
    weighted_sums(1)%out_layer = input
    do i = 2, size(net)+1
        ! Allocating space
        allocate(activations(i)%out_layer(net(i-1)%layer_size))
        allocate(weighted_sums(i)%out_layer(net(i-1)%layer_size))
        ! Weighted sums are calculated; activations are set equal to them (no need to do same calculation twice)
        weighted_sums(i)%out_layer = matmul(activations(i-1)%out_layer, net(i-1)%weights) + net(i-1)%biases
        activations(i)%out_layer = weighted_sums(i)%out_layer
        ! Activations are activated and weighted sums have activation function derivatives done to them
        ! Doing the derivatives now, not in backpropagation - after this step, weighted sums no longer weighted sums
        call activation_func(activations(i)%out_layer, net(i-1)%activation)
        call activation_func_derivative(weighted_sums(i)%out_layer, net(i-1)%activation)
    end do

    ! Back Propagation
    ! Indexing beyond here looks bad but is somehow understandable (thank you FORTRAN for having 1-indexed arrays)

    ! After this step, array "activations" is repurposed step-by-step into their derivatives with repect to the cost
    ! v Changing last layer activations into its derivative v
    call cost_func_derivative(activations(size(activations))%out_layer, goal, net_cost_func)
    do i = size(net), 1, -1
        ! Derivative of biases
        weighted_sums(i+1)%out_layer = activations(i+1)%out_layer*weighted_sums(i+1)%out_layer
        sgd_out(i)%biases = sgd_out(i)%biases + weighted_sums(i+1)%out_layer
        ! Deallocating unnecessary arrays
        deallocate(activations(i+1)%out_layer)
        ! Derivative of weights and activations in last/next (depending on perspective) layer
        do j = 1, size(activations(i)%out_layer)
            sgd_out(i)%weights(j, :) = sgd_out(i)%weights(j, :) + weighted_sums(i+1)%out_layer * activations(i)%out_layer(j)
        end do
        ! After being used, can now be reassigned to its derivative
        activations(i)%out_layer = matmul(weighted_sums(i+1)%out_layer, transpose(net(i)%weights))
        ! Deallocating unnecessary arrays
        deallocate(weighted_sums(i+1)%out_layer)
    end do

    ! Deallocate
    deallocate(activations(1)%out_layer)
    deallocate(weighted_sums(1)%out_layer)
end subroutine sgd

! sgd adds values to sgd_out, sgd_replace replaces the values of sgd_out
subroutine sgd_replace(sgd_out, net, input, goal, net_cost_func)
    implicit none

    type(net_layer), dimension(:), intent(IN) :: net
    type(net_out_layer), dimension(size(net)+1) :: activations, weighted_sums
    real, dimension(:), intent(IN) :: input
    real, dimension(:), intent(IN) :: goal
    character (len = 1), intent(IN) :: net_cost_func
    type(net_layer), dimension(:), intent(INOUT) :: sgd_out
    integer :: i, j

    ! Forward Propagation (a lot of this code was stolen from fwd_prop)

    ! Allocates space and assignes values to first layer of weighted sums and activations
    allocate(activations(1)%out_layer(size(input)))
    allocate(weighted_sums(1)%out_layer(size(input)))
    activations(1)%out_layer = input
    weighted_sums(1)%out_layer = input
    do i = 2, size(net)+1
        ! Allocating space
        allocate(activations(i)%out_layer(net(i-1)%layer_size))
        allocate(weighted_sums(i)%out_layer(net(i-1)%layer_size))
        ! Weighted sums are calculated; activations are set equal to them (no need to do same calculation twice)
        weighted_sums(i)%out_layer = matmul(activations(i-1)%out_layer, net(i-1)%weights) + net(i-1)%biases
        activations(i)%out_layer = weighted_sums(i)%out_layer
        ! Activations are activated and weighted sums have activation function derivatives done to them
        ! Doing the derivatives now, not in backpropagation - after this step, weighted sums no longer weighted sums
        call activation_func(activations(i)%out_layer, net(i-1)%activation)
        call activation_func_derivative(weighted_sums(i)%out_layer, net(i-1)%activation)
    end do

    ! Back Propagation
    ! Indexing beyond here looks bad but is somehow understandable (thank you FORTRAN for having 1-indexed arrays)

    ! After this step, array "activations" is repurposed step-by-step into their derivatives with repect to the cost
    ! v Changing last layer activations into its derivative v
    call cost_func_derivative(activations(size(activations))%out_layer, goal, net_cost_func)
    do i = size(net), 1, -1
        ! Derivative of biases
        weighted_sums(i+1)%out_layer = activations(i+1)%out_layer*weighted_sums(i+1)%out_layer
        sgd_out(i)%biases = weighted_sums(i+1)%out_layer
        ! Deallocating unnecessary arrays
        deallocate(activations(i+1)%out_layer)
        ! Derivative of weights and activations in last/next (depending on perspective) layer
        do j = 1, size(activations(i)%out_layer)
            sgd_out(i)%weights(j, :) = weighted_sums(i+1)%out_layer * activations(i)%out_layer(j)
        end do
        ! After being used, can now be reassigned to its derivative
        activations(i)%out_layer = matmul(weighted_sums(i+1)%out_layer, transpose(net(i)%weights))
        ! Deallocating unnecessary arrays
        deallocate(weighted_sums(i+1)%out_layer)
    end do

    ! Deallocate
    deallocate(activations(1)%out_layer)
    deallocate(weighted_sums(1)%out_layer)
end subroutine sgd_replace

! For use in multiprocessing; does not reset values to zero (use with sgd_replace)
! dimension(network layers, number of batches) for sgd_arr
subroutine update_params_multiple(net, sgd_arr, learning_rate, num_batches)
    implicit none

    type(net_layer), dimension(:), intent(INOUT) :: net
    type(net_layer), dimension(:, :), intent(IN) :: sgd_arr
    real, intent(IN) :: learning_rate
    integer, intent(IN) :: num_batches
    real :: lr
    integer :: i, j
    lr = learning_rate/num_batches

    do i = 1, size(sgd_arr, 1)
        do j = 1, size(sgd_arr, 2)
            net(i)%weights = net(i)%weights - lr * sgd_arr(i, j)%weights
            net(i)%biases = net(i)%biases - lr * sgd_arr(i, j)%biases
        end do
    end do
end subroutine update_params_multiple

! Stolen from last year - update params: weight_new = weight - learning_rate * dC/dweight
subroutine update_params(net, sgd_arr, learning_rate, num_batches)
    implicit none

    type(net_layer), dimension(:), intent(INOUT) :: net, sgd_arr
    real, intent(IN) :: learning_rate
    integer, intent(IN) :: num_batches
    real :: lr
    integer :: i
    lr = learning_rate/num_batches

    do i = 1, size(net)
        net(i)%weights = net(i)%weights - lr * sgd_arr(i)%weights
        net(i)%biases = net(i)%biases - lr * sgd_arr(i)%biases
        sgd_arr(i)%weights = 0
        sgd_arr(i)%biases = 0
    end do
end subroutine update_params

end module FortranNetwork