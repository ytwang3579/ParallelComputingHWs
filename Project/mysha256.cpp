#include "sha256.h"

int main() {
	char buffer[65], buffer1[65], buffer2[65];
	sha256_string("8c14f0db3df150123e6f3dbbf30f8b955a8249b62ac1d1ff16284aefa3d06d87fff2525b8931402dd09222c50775608f75787bd2b87e56995a7bdd30f79702c4", buffer1);
	sha256_string("6359f0868171b1d194cbee1af2f16ea598ae8fad666d9b012c8ed2b79a236ec4e9a66845e05d5abc0ad04ec80f774a7e585c6e8db975962d069a522137b80c1d", buffer2);
	
	printf("%s\n", buffer1);
	printf("%s\n", buffer2);

	sha256_string("5902dbb805b04218174427fd25dcbb36e9569747c6d681037f1d9a471aff38c3118acf2d9cba1f0864e22300f9b6ee0efdacd41357e46e712a9a55bf1c32efa5", buffer);
	printf("%s\n", buffer);

	return 0;
}
