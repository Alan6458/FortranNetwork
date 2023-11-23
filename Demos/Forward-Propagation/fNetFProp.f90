program fNetwork
    use FortranNetwork
    implicit none
    
    integer :: ln(2) = (/2, 3/)
    character (len = 1) :: a
    integer :: i
    type(fNetLayer), dimension(1) :: fn
    type(fNetOutLayer), dimension(size(ln)) :: res
    a = "r"
    fn = initiateNetwork(ln, a)
    fn(1)%weights(1,:) = (/2, 3, 5/)
    fn(1)%weights(2,:) = (/6, 7, 4/)
    fn(1)%biases = (/2, 3, 4/)
    do i = 1, 2
        print *, fn(1)%weights(i,:)
    end do
    print *, ""
    print *, fn(1)%biases
    print *, fn(1)%activation
    print *, fn(1)%layerSize
    print *, ""
    res = forwardProp(fn, (/3, 4/))
    do i = 1, 2
        print *, res(i)%outLayer
    end do
    print *, forwardPropAns(res)
end program fNetwork