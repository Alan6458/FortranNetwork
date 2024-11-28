program fNetwork
    use FortranNetwork
    implicit none
    
    integer :: ln(2) = (/2, 3/)
    character (len = 1) :: a
    integer :: i
    type(net_layer), dimension(1) :: fn
    type(net_out_layer), dimension(size(ln)) :: res
    a = "r"
    call init_net(fn, ln, a)
    fn(1)%weights(1,:) = (/2, 3, 5/)
    fn(1)%weights(2,:) = (/6, 7, 4/)
    fn(1)%biases = (/2, 3, 4/)
    do i = 1, 2
        print *, fn(1)%weights(i,:)
    end do
    print *, ""
    print *, fn(1)%biases
    print *, fn(1)%activation
    print *, fn(1)%layer_size
    print *, ""
    res = fwd_prop(fn, (/3.0, 4.0/))
    do i = 1, 2
        print *, res(i)%out_layer
    end do
    print *, fwd_prop_res(res)
end program fNetwork