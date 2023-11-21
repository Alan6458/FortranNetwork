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

contains

subroutine initiateNetwork(network, layers, activation, layerNodes)
    implicit none

    type(fNet), intent(INOUT) :: network
    integer, intent(IN) :: layers
    integer, dimension(layers), intent(IN) :: layerNodes
    character (len=1), intent(IN) :: activation
    integer i

    ! allocates space for stuff and then fills the variables
    ! Network only needs layers-1 amount of weights/layers
    allocate(network%layer(layers-1))
    allocate(network%layerSizes(layers))
    network%layerSizes = layerNodes
    network%activation = activation
    ! allocating space for each layer of weights and biases
    do i = 1, layers-1
        ! Again, only needs layers-1 amout of weights/layers which is why it is i and i+1 being used
        ! (I predict this may be an indexing nightmare later on)
        allocate(network%layer(i)%weights(layerNodes(i), layerNodes(i+1)))
        allocate(network%layer(i)%biases(layerNodes(i+1)))

        ! generates random floats between 0 and 1 and makes them between -1 and 1
        call random_number(network%layer(i)%weights)
        network%layer(i)%weights = network%layer(i)%weights * 2 - 1

        call random_number(network%layer(i)%biases)
        network%layer(i)%biases = network%layer(i)%biases * 2 - 1
    end do
end subroutine initiateNetwork

end module FortranNetwork