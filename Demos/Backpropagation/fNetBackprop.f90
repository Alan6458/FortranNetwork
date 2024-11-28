program fNetwork
    use FortranNetwork
    implicit none
    
    integer :: ln(2) = (/2, 2/)
    character (len = size(ln)-1) :: a
    character (len = 1) :: c
    integer :: i, j
    type(net_layer), dimension(size(ln)-1) :: fn, fn_sgd
    type(net_out_layer), dimension(size(ln)) :: fn_fwd_prop_out
    real, dimension(2) :: fn_out
    real, dimension(2) :: net_input, net_out_goal
    real :: learn_rate
    integer :: epochs
    integer :: batch_size
    integer :: num_all
    real :: num_correct
    real :: start, finish
    num_all = 100
    num_correct = 0

    call random_seed()

    ! Check start time
    call cpu_time(start)

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
    ! Mini-Batch Size
    batch_size = 200
    ! Learning rate
    learn_rate = 0.5
    ! Activation function(s)
    a = "s"
    ! Cost function
    c = "s"
    net_out_goal = (/0.0001, 0.9999/)
    call init_net(fn, ln, a)
    call init_net(fn_sgd, ln, a)

    ! Sets all values of fn_sgd to zero (updateParams automatically does this)
    do i = 1, size(fn_sgd)
        fn_sgd(i)%weights = 0
        fn_sgd(i)%biases = 0
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
        print *, fn(i)%layer_size
        print *, ""
    end do

    ! Training
    do i = 1, epochs
        ! Stochastic Gradient Descent
        do j = 1, batch_size
            call random_number(net_input)
            call sgd(fn_sgd, fn, net_input, net_out_goal, c)
        end do
        ! Updating weights and biases
        call update_params(fn, fn_sgd, learn_rate, batch_size)

        ! Collecting data
        num_correct = 0
        do j = 1, num_all
            call random_number(net_input)
            fn_fwd_prop_out = fwd_prop(fn, net_input)
            fn_out = fn_fwd_prop_out(size(fn_fwd_prop_out))%out_layer
            call cost_func(fn_out, net_out_goal, c)
            num_correct = num_correct + sum(fn_out)/2
        end do
        ! v Code for showing the actual values of the output v
        ! write(*, 1) fn_fwd_prop_out(size(fn_fwd_prop_out))%out_layer
        ! 1 format(2E12.5)
        ! Printing error of network
        print *, "Epoch #", i, "      Network Error:", num_correct/num_all
    end do

    ! Check finish time
    call cpu_time(finish)

    ! Prints the network's parameters
    print *, ""
    do i = 1, size(fn)
        do j = 1, ln(i)
            print *, fn(i)%weights(j,:)
        end do
        print *, ""
        print *, fn(i)%biases
        print *, fn(i)%activation
        print *, fn(i)%layer_size
        print *, ""
    end do

    ! Prints runtime
    print *, "Time (seconds): ", finish-start
end program fNetwork