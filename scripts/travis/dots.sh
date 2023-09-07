"$@" &

while kill -0 $!; do
    printf '.' > /dev/tty
    sleep 1
done

printf '\n' > /dev/tty
