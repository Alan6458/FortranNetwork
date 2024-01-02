module FortranNetwork
implicit none

type fNetLayer
    ! change from CrapPyNetwork: input layer no longer has biases
    ! Note: this feature may have not been in CrapPyNetwork, I need to check
    real (kind = 4), allocatable, dimension(:,:) :: weights
    real (kind = 4), allocatable, dimension(:) :: biases
    integer :: layerSize
    character :: activation
end type fNetLayer

type fNetOutLayer
    real (kind = 4), allocatable, dimension(:) :: outLayer
end type fNetOutLayer

contains

function initiateNetwork(layerNodes, activation)
    implicit none

    integer, dimension(:), intent(IN) :: layerNodes
    type(fNetLayer), dimension(size(layerNodes)-1) :: initiateNetwork
    character (len=size(layerNodes)-1), intent(IN) :: activation
    integer i

    ! Network only needs layers-1 amount of weights/layers
    ! allocating space for each layer of weights and biases
    do i = 1, size(layerNodes)-1
        ! Again, only needs layers-1 amout of weights/layers which is why it is i and i+1 being used
        ! (I predict this may be an indexing nightmare later on)
        allocate(initiateNetwork(i)%weights(layerNodes(i), layerNodes(i+1)))
        allocate(initiateNetwork(i)%biases(layerNodes(i+1)))

        ! Assigns values to activation function and layer size
        initiateNetwork(i)%activation = activation(i:i)
        initiateNetwork(i)%layerSize = layerNodes(i+1)

        ! generates random floats between 0 and 1 and makes them between -1 and 1
        call random_number(initiateNetwork(i)%weights)
        initiateNetwork(i)%weights = initiateNetwork(i)%weights * 2 - 1

        call random_number(initiateNetwork(i)%biases)
        initiateNetwork(i)%biases = initiateNetwork(i)%biases * 2 - 1
    end do
end function initiateNetwork

function forwardProp(network, input)
    implicit none

    type(fNetLayer), dimension(:), intent(IN) :: network
    real (kind = 4), dimension(:), intent(IN) :: input
    type(fNetOutLayer), dimension(size(network)+1) :: forwardProp
    integer i

    ! The input layer - Should this be included in output?
    ! Again, this may be an indexing issue later on
    allocate(forwardProp(1)%outLayer(size(input)))
    forwardProp(1)%outLayer = input
    ! Allocates space for next layer and performs matrix multiplication and adds biases, then does activation
    do i = 2, size(network)+1
        ! Here's (just the beginning of) the indexing nightmare I predicted earlier
        allocate(forwardProp(i)%outLayer(network(i-1)%layerSize))
        forwardProp(i)%outLayer = matmul(forwardProp(i-1)%outLayer, network(i-1)%weights) + network(i-1)%biases
        deallocate(forwardProp(i-1)%outLayer)
        call activationFunction(forwardProp(i)%outLayer, network(i-1)%activation)
    end do
end function forwardProp

! This outputs the index of the max output node of an fNetOutLayer object
function forwardPropAns(fPropOut)
    implicit none
    
    type(fNetOutLayer), dimension(:), intent(IN) :: fPropOut
    integer, dimension(1) :: preAns
    integer forwardPropAns
    preAns = maxloc(fPropOut(size(fPropOut))%outLayer)
    forwardPropAns = preAns(1)
end function forwardPropAns

! Note: this is just temporary, may need a faster method
! Need to have function as a variable in the layer type instead of char, like lambda in Python
subroutine activationFunction(nodeVals, activationType)
    implicit none

    character (len = 1), intent(IN) :: activationType
    real (kind = 4), dimension(:), intent(INOUT) :: nodeVals
    
    if (activationType == "s") then
        ! Sigmoid
        nodeVals = 1/(1+exp(-nodeVals))
    else if (activationType == "t") then
        ! TanH
        nodeVals = tanh(nodeVals)
    else if (activationType == "r") then
        ! ReLU
        ! nodeVals = (nodeVals + abs(nodeVals))/2
        where (nodeVals < 0.0) nodeVals = 0.0
    else if (activationType == "S") then
        ! SiLU
        nodeVals = nodeVals/(1+exp(-nodeVals))
    else if (activationType == "m") then
        ! Softmax
        nodeVals = exp(nodeVals)/sum(exp(nodeVals))
    end if
end subroutine activationFunction

subroutine activationFunctionDerivative(nodeVals, activationType)
    implicit none

    character (len = 1), intent(IN) :: activationType
    real (kind = 4), dimension(:), intent(INOUT) :: nodeVals
    
    ! This is like activation function but the derivative (for backprop)

    if (activationType == "s") then
        ! Sigmoid
        nodeVals = exp(-nodeVals)/((1+exp(-nodeVals))**2)
    else if (activationType == "t") then
        ! TanH
        nodeVals = 1-(tanh(nodeVals)**2)
    else if (activationType == "r") then
        ! ReLU
        ! nodeVals = (nodeVals/abs(nodeVals)+1)/2
        where (nodeVals >= 0.0) nodeVals = 1.0
        where (nodeVals < 0.0) nodeVals = 0.0
    else if (activationType == "S") then
        ! SiLU
        nodeVals = (1+exp(-nodeVals)*(1+nodeVals))/((1+exp(-nodeVals))**2)
    else if (activationType == "m") then
        ! Softmax (this is gonna be slow but whatever)
        nodeVals = (exp(nodeVals)/sum(exp(nodeVals)))*(1-(exp(nodeVals)/sum(exp(nodeVals))))
    end if
end subroutine activationFunctionDerivative

subroutine costFunction(nodeVals, goal, costType)
    implicit none

    character (len = 1), intent(IN) :: costType
    real (kind = 4), dimension(:), intent(IN) :: goal
    real (kind = 4), dimension(:), intent(INOUT) :: nodeVals

    if (costType == "s") then
        ! Mean Squared Error (MSE)
        nodeVals = (nodeVals-goal)**2
    else if (costType == "a") then
        ! Mean Absolute Error (MAE)
        nodeVals = abs(nodeVals-goal)
    else if (costType == "c") then
        ! Softmax Cross-Entropy Loss
        nodeVals = -goal*log(exp(nodeVals)/sum(exp(nodeVals)))
    end if
end subroutine costFunction

subroutine costFunctionDerivative(nodeVals, goal, costType)
    implicit none

    character (len = 1), intent(IN) :: costType
    real (kind = 4), dimension(:), intent(IN) :: goal
    real (kind = 4), dimension(:), intent(INOUT) :: nodeVals

    if (costType == "s") then
        ! Mean Squared Error (MSE)
        nodeVals = 2 * (nodeVals-goal)
    else if (costType == "a") then
        ! Mean Absolute Error (MAE)
        ! nodeVals = nodeVals/abs(nodeVals)
        where (nodeVals >= 0.0) nodeVals = 1.0
        where (nodeVals < 0.0) nodeVals = -1.0
    else if (costType == "c") then
        ! Softmax Cross-Entropy Loss
        nodeVals = exp(nodeVals)/sum(exp(nodeVals))-goal
    end if
end subroutine costFunctionDerivative

subroutine sgd(sgdOut, fnet, input, goal, costFunc)
    implicit none

    type(fNetLayer), dimension(:), intent(IN) :: fnet
    type(fNetOutLayer), dimension(size(fnet)+1) :: activations, weightedSums
    real (kind = 4), dimension(:), intent(IN) :: input
    real (kind = 4), dimension(:), intent(IN) :: goal
    character (len = 1), intent(IN) :: costFunc
    type(fNetLayer), dimension(:), intent(INOUT) :: sgdOut
    integer :: i, j

    ! Forward Propagation (a lot of this code was stolen from forwardProp)

    ! Allocates space and assignes values to first layer of weighted sums and activations
    allocate(activations(1)%outLayer(size(input)))
    allocate(weightedSums(1)%outLayer(size(input)))
    activations(1)%outLayer = input
    weightedSums(1)%outLayer = input
    do i = 2, size(fnet)+1
        ! Allocating space
        allocate(activations(i)%outLayer(fnet(i-1)%layerSize))
        allocate(weightedSums(i)%outLayer(fnet(i-1)%layerSize))
        ! Weighted sums are calculated; activations are set equal to them (no need to do same calculation twice)
        weightedSums(i)%outLayer = matmul(activations(i-1)%outLayer, fnet(i-1)%weights) + fnet(i-1)%biases
        activations(i)%outLayer = weightedSums(i)%outLayer
        ! Activations are activated and weighted sums have activation function derivatives done to them
        ! Doing the derivatives now, not in backpropagation - after this step, weighted sums no longer weighted sums
        call activationFunction(activations(i)%outLayer, fnet(i-1)%activation)
        call activationFunctionDerivative(weightedSums(i)%outLayer, fnet(i-1)%activation)
    end do

    ! Back Propagation
    ! Indexing beyond here looks bad but is somehow understandable (thank you FORTRAN for having 1-indexed arrays)

    ! After this step, array "activations" is repurposed step-by-step into their derivatives with repect to the cost
    ! v Changing last layer activations into its derivative v
    call costFunctionDerivative(activations(size(activations))%outLayer, goal, costFunc)
    do i = size(fnet), 1, -1
        ! Derivative of biases
        sgdOut(i)%biases = activations(i+1)%outLayer*weightedSums(i+1)%outLayer
        ! Deallocating unnecessary arrays
        deallocate(weightedSums(i+1)%outLayer)
        deallocate(activations(i+1)%outLayer)
        ! Derivative of weights and activations in last/next (depending on perspective) layer
        do j = 1, size(activations(i)%outLayer)
            sgdOut(i)%weights(j, :) = sgdOut(i)%biases * fnet(i)%weights(j, :) * activations(i)%outLayer(j)
            ! After being used, can now be reassigned to its derivative
            activations(i)%outLayer(j) = sum(sgdOut(i)%weights(j, :))
        end do
    end do
end subroutine sgd

! Stolen from last year - update params: weight_new = weight - learningRate * dC/dweight
subroutine updateParams(fnet, sgdArr, learningRate)
    implicit none

    type(fNetLayer), dimension(:), intent(INOUT) :: fnet
    type(fNetLayer), dimension(:, :), intent(IN) :: sgdArr
    type(fNetLayer), dimension(size(fnet)) :: tempArr
    real (kind = 4), intent(IN) :: learningRate
    real (kind = 4) :: lR
    integer :: i, j
    lR = learningRate/size(sgdArr, DIM = 1)

    do i = 1, size(fnet)
        allocate(tempArr(i)%weights(size(fnet(i)%weights, DIM = 1), fnet(i)%layerSize))
        allocate(tempArr(i)%biases(fnet(i)%layerSize))
        do j = 1, size(sgdArr, DIM = 1)
            tempArr(i)%weights = tempArr(i)%weights + sgdArr(j, i)%weights
            tempArr(i)%biases = tempArr(i)%biases + sgdArr(j, i)%biases
        end do
        fnet(i)%weights = fnet(i)%weights - lR * tempArr(i)%weights
        fnet(i)%biases = fnet(i)%biases - lR * tempArr(i)%biases
    end do
end subroutine updateParams

end module FortranNetwork