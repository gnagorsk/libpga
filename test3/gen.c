/*
 * gen.c
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int get_rand(int min, int max) {
	return rand() % max + min;
}

int main() {
	srand(time(NULL));
	printf("100\n");
	for (int i = 0; i < 100; ++i) {
		for(int j = 0; j < 100; ++j) {
			if (i == (j-1)) {
				printf("%u ", 10);
			} else {
				printf("%u ", get_rand(10, 1000));
			}
			
		}
		printf("\n");
	}
	
	return 0;
}