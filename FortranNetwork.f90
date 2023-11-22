module FortranNetwork
implicit none

type fNet
    type(layers), allocatable, dimension(:) :: layer
    integer, allocatable, dimension(:) :: layerSizes
    character (len = 1) :: activation
end type fNet

type layers
    ! change from CrapPyNetwork: input layer no longer has biases
    ! Note: this feature may have not been in CrapPyNetwork, I need to check
    real (kind = 4), allocatable, dimension(:,:) :: weights
    real (kind = 4), allocatable, dimension(:) :: biases
end type layers

type fNetOutLayer
    real (kind = 4), allocatable, dimension(:) :: outLayer
end type fNetOutLayer

contains

function initiateNetwork(layerNodes, activation)
    implicit none

    type(fNet) :: initiateNetwork
    integer, dimension(:), intent(IN) :: layerNodes
    character (len=1), intent(IN) :: activation
    integer :: layers
    integer i
    layers = size(layerNodes)

    ! allocates space for stuff and then fills the variables
    ! Network only needs layers-1 amount of weights/layers
    allocate(initiateNetwork%layer(layers-1))
    allocate(initiateNetwork%layerSizes(layers))
    initiateNetwork%layerSizes = layerNodes
    initiateNetwork%activation = activation
    ! allocating space for each layer of weights and biases
    do i = 1, layers-1
        ! Again, only needs layers-1 amout of weights/layers which is why it is i and i+1 being used
        ! (I predict this may be an indexing nightmare later on)
        allocate(initiateNetwork%layer(i)%weights(layerNodes(i), layerNodes(i+1)))
        allocate(initiateNetwork%layer(i)%biases(layerNodes(i+1)))

        ! generates random floats between 0 and 1 and makes them between -1 and 1
        call random_number(initiateNetwork%layer(i)%weights)
        initiateNetwork%layer(i)%weights = initiateNetwork%layer(i)%weights * 2 - 1

        call random_number(initiateNetwork%layer(i)%biases)
        initiateNetwork%layer(i)%biases = initiateNetwork%layer(i)%biases * 2 - 1
    end do
end function initiateNetwork

function forwardProp(network, input)
    implicit none

    type(fNet), intent(IN) :: network
    integer, dimension(:), intent(IN) :: input
    type(fNetOutLayer), dimension(size(network%layerSizes)) :: forwardProp
    integer i

    ! The input layer - Should this be included in output?
    ! Again, this may be an indexing issue later on
    allocate(forwardProp(1)%outLayer(network%layerSizes(1)))
    forwardProp(1)%outLayer = input
    ! Allocates space for next layer and then performs matrix multiplication and adds biases
    do i = 2, size(network%layerSizes)
        ! Here's (just the beginning of) the indexing nightmare I predicted earlier
        allocate(forwardProp(i)%outlayer(network%layerSizes(i)))
        forwardProp(i)%outlayer = matmul(forwardProp(i-1)%outLayer, network%layer(i-1)%weights) + network%layer(i-1)%biases
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

end module FortranNetwork