# cpp-neural-network
Basic neural network implemented in C++, visualized with SFML. Currently predicts pixel color based on a few given inputs. Added multithreading with OpenMP.

To run the application, download the Debug folder with the .exe and all the .dlls, and run neuralnetwork.exe

To get open the solution, you need to set the additional include libraries path and additioan library dependencies path in order for the program to use the SFML libraries.

To do this:
1. Right click on the project "Neural Network" in the seolution explorer tab
2. Click "Properties"
3. Navigate to Linker > General > Additional Library Directories
4. Click the drop down arrow on the right, and click edit
5. Double click the path that's already there, and click the button with three dots that apperas to the right
6. Navigate to the project folder, then navigate to sfml > lib, click select folder, click OK
7. Now navigate to C/C++ > General > Additional Include Directories
8. Repeat steps 4-6, except in step 6 navigate to sfml > include instead
9. Click apply, click OK
10. The project should build and run successfully now.
