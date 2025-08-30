#include <stdio.h>

float average(int a, int b, int c, int d, int e, int f);
void grading(float avg);

int main() {
    int a, b, c, d, e, f;

    printf("Enter your marks for Maths: ");
    scanf("%d", &a);

    printf("Enter your marks for Physics: ");
    scanf("%d", &b);

    printf("Enter your marks for Chemistry: ");
    scanf("%d", &c);

    printf("Enter your marks for English: ");
    scanf("%d", &d);

    printf("Enter your marks for Computer Science: ");
    scanf("%d", &e);

    printf("Enter your marks for Physical Education: ");
    scanf("%d", &f);

    float avg = average(a, b, c, d, e, f); 
    grading(avg);                           

    return 0;
}

float average(int a, int b, int c, int d, int e, int f) {
    float avg = (a + b + c + d + e + f) / 6.0;   
    printf("The average is: %f\n", avg);       
    return avg;
}

void grading(float avg) {
    if (avg > 85 && avg <= 100)
        printf("Grade: A+\n");
    else if (avg > 75 && avg <= 85)
        printf("Grade: A\n");
    else if (avg > 55 && avg <= 75)
        printf("Grade: B+\n");
    else if (avg > 45 && avg <= 55)
        printf("Grade: B\n");
    else if (avg >= 35 && avg <= 45)
        printf("Grade: C\n");
    else if (avg >= 25 && avg < 35)
        printf("Grade: C+\n");
    else if (avg >= 0 && avg < 25)
        printf("Grade: W\n");
    else
        printf("Invalid Marks\n");
}
