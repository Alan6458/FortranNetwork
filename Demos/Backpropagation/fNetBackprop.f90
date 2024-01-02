program fNetwork
    use FortranNetwork
    implicit none
    
    integer :: ln(2) = (/2, 2/)
    character (len = size(ln)-1) :: a
    character (len = 1) :: c
    integer :: i, j
    type(fNetLayer), dimension(size(ln)-1) :: fn
    type(fNetLayer), dimension(200, size(ln)-1) :: fnSGD
    type(fNetOutLayer), dimension(size(ln)) :: fnFOut
    real (kind = 4), dimension(2) :: fnOut
    real (kind = 4), dimension(2) :: netInput, netOutputGoal
    real (kind = 4) :: learnRate
    integer :: epochs
    integer :: batchSize = size(fnSGD, dim = 1)
    integer :: numAll
    real :: numCorrect
    numAll = 100
    numCorrect = 0

    call random_seed()

    ! Activation Functions
    ! --------------------
    ! s - Sigmoid
    ! t - TanH
    ! r - ReLU
    ! S - SiLU/Swish
    ! m - Softmax

    ! Cost Functions
    ! --------------
    ! s - Mean Squared Error (MSE)
    ! a - Mean Absolute Error (MAE)
    ! c - Softmax Cross-Entropy Loss


    ! Epochs to train for
    epochs = 20
    ! Learning rate
    learnRate = 0.5
    ! Activation function(s)
    a = "s"
    ! Cost function
    c = "s"
    netOutputGoal = (/0.0001, 0.9999/)
    fn = initiateNetwork(ln, a)

    do i = 1, batchSize
        fnSGD(i, :) = fn
    end do

    ! Prints the network's parameters
    print *, ""
    do i = 1, size(fn)
        do j = 1, ln(i)
            print *, fn(i)%weights(j,:)
        end do
        print *, ""
        print *, fn(i)%biases
        print *, fn(i)%activation
        print *, fn(i)%layerSize
        print *, ""
    end do

    ! Training
    do i = 1, epochs
        ! Stochastic Gradient Descent
        do j = 1, batchSize
            call random_number(netInput)
            call sgd(fnSGD(j, :), fn, netInput, netOutputGoal, c)
        end do
        ! Updating weights and biases
        call updateParams(fn, fnSGD, learnRate)

        ! Collecting data
        numCorrect = 0
        do j = 1, numAll
            call random_number(netInput)
            fnFOut = forwardProp(fn, netInput)
            fnOut = fnFOut(size(fnFOut))%outLayer
            call costFunction(fnOut, netOutputGoal, c)
            numCorrect = numCorrect + sum(fnOut)/2
        end do
        ! v Code for showing the actual values of the output v
        ! write(*, 1) fnFOut(size(fnFOut))%outLayer
        ! 1 format(2E12.5)
        ! Printing error of network
        print *, "Epoch #", i, "      Network Error:", numCorrect/numAll
    end do

    ! Prints the network's parameters
    print *, ""
    do i = 1, size(fn)
        do j = 1, ln(i)
            print *, fn(i)%weights(j,:)
        end do
        print *, ""
        print *, fn(i)%biases
        print *, fn(i)%activation
        print *, fn(i)%layerSize
        print *, ""
    end do
end program fNetwork