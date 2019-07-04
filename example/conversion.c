int main(void) {
	int flag, array[30], b = 3;

	if (flag)
		goto done;

	int (* M)[b] = (int (*)[b]) array;

done:
	return 0;
}