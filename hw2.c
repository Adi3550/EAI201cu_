#include <stdio.h>
int main() {
    int command;
    int shape_choice;
    char *shape;
    char *dustType;


    printf("Select Vacuum Shape:\n");
    printf("1. Circle  (Best for Fine Dust)\n");
    printf("2. Square  (Best for Large Debris)\n");
    printf("3. Triangle (Best for Liquid Spills)\n");
    printf("4. Hexagon (Best for Mixed Cleaning)\n");
    printf("Enter your choice: ");
    scanf("%d", &shape_choice);

    switch(shape_choice) {
        case 1:
            shape = "Circle";
            dustType = "Fine Dust";
            break;
        
        case 2:
            shape = "Square";
            dustType = "Large Debris";
            break;
        
        case 3:
            shape = "Triangle";
            dustType = "Liquid Spills";
            break;
        
        case 4:
            shape = "Hexagon";
            dustType = "Mixed Cleaning";
            break;
        
            default:
            printf("Invalid shape choice!\n");
            return 0;
    }

    printf("\n🌀 Shape: %s\n", shape);
    printf("🧹 Best for: %s\n", dustType);
    printf("----------------------------------------\n");

    
    
     while (1) {
        printf("\nCommands:\n");
        printf("1. Start\n");
        printf("2. Stop\n");
        printf("3. Left\n");
        printf("4. Right\n");
        printf("5. Dock\n");
        printf("6. Exit\n");

        printf("Enter your command number: ");
        scanf("%d", &command);

        switch(command) {
            case 1:
                printf("[START] The %s-shaped vacuum starts cleaning %s.\n", shape, dustType);
                break;
            case 2:
                printf("[STOP] The %s-shaped vacuum has stopped.\n", shape);
                break;
            case 3:
                printf("[MOVE] Turning LEFT to cover more %s.\n", dustType);
                break;
            case 4:
                printf("[MOVE] Turning RIGHT to cover more %s.\n", dustType);
                break;
            case 5:
                printf("[DOCK] Returning to docking station. Battery recharging...\n");
                break;
            case 6:
                printf("[EXIT] Shutting down vacuum program.\n");
                return 0; 
            default:
                printf("Invalid command! Please try again.\n");
        }

     }
    
    return 0;

}
