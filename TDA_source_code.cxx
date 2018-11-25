    <Topological Data Analysis algorithm for generating topological feature descriptors of atmospheric rivers.
    This work was supported by Intel Parallel Computing Center at University of Liverpool, UK and funded by Intel.>
    Copyright (C) <2018>  <Grzegorz Muszynski>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <iostream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <chrono>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cassert>

using std::cerr;
using std::endl;

class Point 
{
    private:
        double xval, yval;
    public:
        // Constructor uses default arguments to allow calling with zero, one,
        // or two values.
        Point(double x = 0.0, double y = 0.0) {
            xval = x;
            yval = y;
        }
    
        // Extractors.
        double x() { return xval; }
        double y() { return yval; }
};

//Class that represents vertices in a graph.
class Node
{
    public:
        int index;
        Point pixel;
        float value;
        int parent = -1, component = -1; // default value for a non-existing node
        unsigned long grid_x;
        unsigned long grid_y;

        //Additional members.
        std::map<float, int> four_neighbors;

        Node(int i, float v, Point p, unsigned long g_x, unsigned long g_y) { index = i, value = v, pixel = p, grid_x = g_x, grid_y = g_y; }

        Node(int i, float v, Point p, std::map<float, int> m, unsigned long g_x, unsigned long g_y) { index = i, value = v, pixel = p, four_neighbors = m, grid_x = g_x, grid_y = g_y; }
};

class Component
{
    public:
        int index;
        unsigned long nb_nodes;
        int max_value;
        int min_value = -1;
        int sum_nodes;
        std::vector<int> inodes;
        std::vector<int> icomponents;

        std::vector<Point> source_nodes;
        std::vector<Point> target_nodes;

        bool region_A;
        bool region_B;

        bool is_tracked;

        int max_value_AB = -1;

        std::map<float, int> neighbors;

        Component(Node node)
        {
            index = node.component;
            nb_nodes = 1;
            max_value = node.value;
            sum_nodes = sum_nodes + node.value;
            region_A = false;
            region_B = false;
            is_tracked = false;

            neighbors = node.four_neighbors;
        }

        Component(Component &c0, Component &c1)
        {
            region_A = false;
            region_B = false;

            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            max_value = std::max(c0.max_value, c1.max_value);

            int i0 = c0.index;
            int i1 = c1.index;

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); //preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );

            source_nodes.reserve( c0.source_nodes.size() + c1.source_nodes.size() );
            source_nodes.insert( source_nodes.end(), c0.source_nodes.begin(), c0.source_nodes.end() );         
            source_nodes.insert( source_nodes.end(), c1.source_nodes.begin(), c1.source_nodes.end() );    
         
         	target_nodes.reserve( c0.target_nodes.size() + c1.target_nodes.size() );
            target_nodes.insert( target_nodes.end(), c0.target_nodes.begin(), c0.target_nodes.end() );         
            target_nodes.insert( target_nodes.end(), c1.target_nodes.begin(), c1.target_nodes.end() );      
        }

        Component(Component &c0, Component &c1, int i)
        {
            region_A = false;
            region_B = false;

            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            max_value = std::max(c0.max_value, c1.max_value);

            index = i;

            int i0 = c0.index;
            int i1 = c1.index;

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); //preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );
        }

        //Merge two old components into a new component.
        Component(int i, float value, Component &c0, Component &c1)
        {
            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            sum_nodes = c0.sum_nodes + c1.sum_nodes;

            max_value = std::max(c0.max_value, c1.max_value);

            int i0 = c0.index;
            int i1 = c1.index;

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); //preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );

            //Checking merging two components for two regions constraint.
            if(c0.region_A || c1.region_A)
            {
                region_A = true;
            }

            if(c0.region_B || c1.region_B)
            {
                region_B = true;
            }

            //Check if both components (c0 and c1) touch both boxes.
            bool c0_both = c0.region_A && c0.region_B;
            bool c1_both = c1.region_A && c1.region_B;

            //If c0 and c1 touches both boxes.
            if(c0_both && c1_both)
            {
                if(c0.max_value_AB > c1.max_value_AB)
                {
                    index = c0.index;
                    max_value_AB = c0.max_value_AB;
                }
                else
                {
                    index = c1.index;
                    max_value_AB = c1.max_value_AB;
                }
            }
            //If c0 and c1 does not touch both boxes.
            else if(!c0_both && !c1_both)
            {
                if(c0.region_A && !c0.region_B && !c1.region_A && c1.region_B)
                {
                    max_value_AB = value; //current value
                    index = i;
                }
                else if(!c0.region_A && c0.region_B && c1.region_A && !c1.region_B)
                {
                    max_value_AB = value; //current value
                    index = i;
                }
                else
                {
                    index = i;
                }
            }
            //If c0 does not touch both boxes, but c1 touches both.
            else if (!c0_both && c1_both)
            {
                index = c1.index;
                max_value_AB = c1.max_value_AB;
            }
            //If c0 touches both boxes, but c0 does not touch both.
            else if (c0_both && !c1_both)
            {
                index = c0.index;
                max_value_AB = c0.max_value_AB;
            }
        }
};

//  Class volume that creates objects with information about nubmer of nodes and threshold value.
class Volume
{
public:
    int v_nb_nodes;
    int v_threshold_value;
    int v_id;
    std::vector<int> v_icomponents;

    Volume(int nb_nodes, int threshold_value)
    {
        v_nb_nodes = nb_nodes;
        v_threshold_value = threshold_value;
    }

    Volume(int nb_nodes, int threshold_value, int id)
    {
        v_nb_nodes = nb_nodes;
        v_threshold_value = threshold_value;
        v_id = id;
    }

    Volume(int nb_nodes, int threshold_value, int id, std::vector<int> icomponents)
    {
        v_nb_nodes = nb_nodes;
        v_threshold_value = threshold_value;
        v_id = id;
        v_icomponents = icomponents;
    }
};

int find_root(std::vector<Node>& nodes, int n0);

void finding_components(std::vector<Node> nodes, std::vector<Edge> edges, std::vector<Component>& components, unsigned long time_step, unsigned int *regionA, unsigned int *regionB);

//Functions for finding threshold parameter.
template <typename T>
    void union_find_alg(const T *input, unsigned long num_rc,
                        unsigned long num_rows, unsigned long num_cols, const_p_teca_variant_array lat,
                        const_p_teca_variant_array lon, unsigned long time_step, const_p_teca_variant_array land_sea_mask);

//  Save volume information to text file.
void save_volume_to_file(std::vector<Volume> volumes, unsigned long time_step)
{
    std::ostringstream oss;
    oss << "ar_volumes_" << std::setw(6) << std::setfill('0') << time_step << ".txt";

    std::string file_name = oss.str();

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    for (std::vector<Volume>::iterator it = volumes.begin(), end = volumes.end(); it != end; ++it)
    {
        ofs << (*it).v_threshold_value << "," << (*it).v_nb_nodes << "," << (*it).v_id;

        if((*it).v_icomponents.size() > 0)
        {
            ofs << "," << (*it).v_icomponents[0] << "," << (*it).v_icomponents[1] << endl;
        }
        else
        {
            ofs << "," << -1 << "," << -1 << endl;
        }
    }

    ofs.close();
}

//  Sorting nodes based on a member value of the class Node.
bool sort_nodes(Node const& n0, Node const& n1)
{
    if(n0.value > n1.value)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//  Creating each node with 4 neighbours of each node and putting it in a vector of nodes.
template <typename T, typename T1>
    void finding_neighbours_of_nodes(std::vector<Node>& nodes1, int num_rows, int num_cols, const T *input, const T1 *p_lon, const T1 *p_lat)
{
    //std::cerr << "func: finding_neighbours_of_nodes() starts..." << endl;
    
    for(int i=0, y=0; y<num_rows; y++)
    {
        for(int x=0; x<num_cols; x++, i++)
        {
            std::map<float, int> neighbours;

            if(x > 0) { neighbours.insert(std::pair<float, int>(input[i - 1], i - 1)); } // left

            if(x < num_cols-1) { neighbours.insert(std::pair<float, int>(input[i + 1], i + 1)); } // right

            if(y > 0) { neighbours.insert(std::pair<float, int>(input[i - num_cols], i - num_cols)); } // above

            if(y < num_rows-1) { neighbours.insert(std::pair<float, int>(input[i + num_cols], i + num_cols)); } // below

            nodes1.push_back( Node(i ,input[i], Point(p_lon[x] , p_lat[y]), neighbours, x, y) );
        }
    }
}

//Save to file points that belong to components that touch box boxes.
void save_components_touching_both_boxes(std::vector<Point>& circumcenters_coordinates, unsigned long time_step, int val, int ind, int nb_n)
{
 
    std::ostringstream oss;
    oss << "ar_comp_"
       << val << "_" << time_step << "_" << ind  << "_" << nb_n << ".vtk";

    std::string file_name = oss.str();

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    unsigned int size_value = 0;

    //If number of points is even.
    if((int)circumcenters_coordinates.size() % 2 == 0)
    {
        size_value =  (int)circumcenters_coordinates.size();
    }
    else //If it is odd.
    {
        size_value = (int)circumcenters_coordinates.size() - 1;
    }

    ofs << "# vtk DataFile Version 4.0" << endl;
    ofs << "vtk output" << endl;
    ofs << "ASCII" << endl;
    ofs << "DATASET UNSTRUCTURED_GRID" << endl;
    ofs << "POINTS " << size_value << " " << "float" << endl;

    ofs << std::setprecision(2) << std::fixed;

    //Set cooridnates of points.
    for(unsigned long p = 0; p < size_value; p+=2)
    {
        ofs << (float)circumcenters_coordinates[p].x() << " " << (float)circumcenters_coordinates[p].y() << " " << 0 << " "
        << (float)circumcenters_coordinates[p + 1].x() << " " << (float)circumcenters_coordinates[p + 1].y() << " " << 0 << " " << endl;
    }

    //Set type of cells.
    int nb_points = 2;

    ofs << "CELLS " << size_value/2 << " " << size_value/2 * (nb_points + 1) << endl;

    for(unsigned long c = 0; c < size_value; c+=2)
    {
        ofs << 2 << " " << c << " " << c + 1 << endl;
    }

    ofs << endl;

    ofs << "CELL_TYPES" << " " << size_value/2 << endl;

    //Set cell type as a number.
    const int cell_type = 1;

    for(unsigned long ct = 0; ct < size_value/2; ct++)
    {
        ofs << cell_type << endl;
    }

    ofs.close();
}

//  Gather information about volume of connected components touching both regions into vector of volumes.
void tracking_volume_of_component(unsigned long time_step, int val,
        Component& component, std::vector<Volume>& volumes, bool& tracking_flag,
        std::vector<Node>& nodes)
{
    //std::cerr << "func: tracking_volume_of_component() starts..." << time_step << endl;

    if(tracking_flag)
    {
        //Save if component touches both regions.
        if(component.region_A && component.region_B)
        {
            volumes.push_back( Volume(component.nb_nodes, val, component.index, component.icomponents) );

            //Save points to file.
            std::vector<Point> selected_points;
            std::vector<float> iwv_values;

            for(unsigned int j = 0; j < component.inodes.size(); ++j)
            {
                Point p = Point(nodes[component.inodes[j]].pixel.x(), nodes[component.inodes[j]].pixel.y());

                selected_points.push_back(p);

                iwv_values.push_back(nodes[component.inodes[j]].value);
            }
            
            save_components_touching_both_boxes(selected_points, time_step, val, component.index, component.nb_nodes);

            //Save only first connected component that satisfy criteria.
            tracking_flag = false;
        }
    }
    else
    {
        volumes.push_back( Volume(component.nb_nodes, val, component.index, component.icomponents) );
    }
}

// Checking if a node is in region A or in region B.
void check_two_region_constraint(int n0, unsigned int *regionA, unsigned int *regionB, Component& c_tmp, std::vector<Node>& nodes)
{
    if(regionA[n0] == 1)
    {
        c_tmp.region_A = true;
        c_tmp.source_nodes.push_back( nodes[n0].pixel );
    }

    if(regionB[n0] == 1)
    {
        c_tmp.region_B = true;
        c_tmp.target_nodes.push_back( nodes[n0].pixel );
    }
}

//  Checking if both connected components touch region A and region B.
void check_when_merging(Component& new_comp, Component& c0, Component& c1, float value, int i)
{
    //Check if both components (c0 and c1) touch both boxes.
    bool c0_both = c0.region_A && c0.region_B;
    bool c1_both = c1.region_A && c1.region_B;

    //If c0 and c1 touches both boxes.
    if(c0_both && c1_both)
    {
        if(c0.max_value_AB > c1.max_value_AB)
        {
            new_comp.index = c0.index;
            new_comp.max_value_AB = c0.max_value_AB;
        }
        else
        {
            new_comp.index = c1.index;
            new_comp.max_value_AB = c1.max_value_AB;
        }

        new_comp.region_A = true;
        new_comp.region_B = true;
    }
    //If c0 and c1 does not touch both boxes.
    else if(!c0_both && !c1_both)
    {
        if(c0.region_A && !c0.region_B && !c1.region_A && c1.region_B)
        {
            new_comp.max_value_AB = value; //current value
            new_comp.index = i;

            new_comp.region_A = true;
            new_comp.region_B = true;
        }
        else if(!c0.region_A && c0.region_B && c1.region_A && !c1.region_B)
        {
            new_comp.max_value_AB = value; //current value
            new_comp.index = i;

            new_comp.region_A = true;
            new_comp.region_B = true;
        }
        else
        {
            new_comp.index = i;

            //If touches one of the boxes.
            if(c0.region_A || c1.region_A)
            {
                new_comp.region_A = true;
            }

            if(c0.region_B || c1.region_B)
            {
                new_comp.region_B = true;
            }
        }
    }
    //If c0 does not touch both boxes, but c1 touches both.
    else if (!c0_both && c1_both)
    {
        new_comp.index = c1.index;
        new_comp.max_value_AB = c1.max_value_AB;

        new_comp.region_A = true;
        new_comp.region_B = true;
    }
    //If c0 touches both boxes, but c0 does not touch both.
    else if (c0_both && !c1_both)
    {
        new_comp.index = c0.index;
        new_comp.max_value_AB = c0.max_value_AB;

        new_comp.region_A = true;
        new_comp.region_B = true;
    }
    else
    {
        std::cerr << "DEBUG" << endl;
    }
}

//  Finding root (parent) of connected component using a recurisve functon.
int find_root(std::vector<Node>& nodes, int n0)
{
    //If node is new.
    if(nodes[n0].parent == -1)
    {
        return -1;
    }
    
    //If n0 is the root.
    if(nodes[n0].parent != n0)
    {
        nodes[n0].parent = find_root(nodes, nodes[n0].parent);
    }
    
    return nodes[n0].parent;
}

template <typename T, typename T1>
    void finding_components_test(std::vector<Component>& components, unsigned long time_step, unsigned int *regionA, unsigned int *regionB, int num_cols, int num_rows, const T *input, const T1 *p_lon, const T1 *p_lat)
{
    std:: cerr << "TIMESTEP -> " << time_step << endl;

    std::vector<Volume> volumes;
    
    std::vector<Node> nodes; //Unsorted list of nodes.
    std::vector<Node> sorted_nodes; //Sorted list of nodes.

    //Creating nodes and finding theirs four neighbours.
    finding_neighbours_of_nodes(nodes, num_rows, num_cols, input, p_lon, p_lat);

    //Make a copy of the vector.
    sorted_nodes = nodes;

    //Sort nodes in the decreasing order.
    std::sort( sorted_nodes.begin(), sorted_nodes.end(), sort_nodes );

    int n_nodes = sorted_nodes.size();

    bool tracking_flag = true;
    
    int n0;
    int n1;
    float val_n1;
    float val;
    int r0;
    int r1;
    int c0;
    int c1;
    
    //Make-set.
    for(int n = 0; n < n_nodes; n++)
    {        
        //Get real index of node.
        n0 = sorted_nodes[n].index;
        
        //Component has been already created.
        if(nodes[n0].parent != -1) 
            continue;

        //It is a root of itself.
        nodes[n0].parent = n0;
        
        //Store index of a new component.
        nodes[n0].component = (int)components.size();
        
        //Create new component.
        Component comp(nodes[n0]);
        
        //Add index of node in a component.
        comp.inodes.push_back(n0);
        
        if(tracking_flag)
            check_two_region_constraint(n0, regionA, regionB, comp, nodes);
        
        //Add component to the list of components.
        components.push_back(comp);
        
        //For each neighbour of node n.
        for (std::map<float, int>::iterator it = nodes[n0].four_neighbors.begin(); it!=nodes[n0].four_neighbors.end(); ++it)
        {
            //Get index of neighbour.
            n1 = it->second;

            //Get value.
            val_n1 = it->first;
            
            //Weight of edge between n0 and its neighbour.
            val = std::min(nodes[n0].value, val_n1);
            
            //Skip lower nodes.
            if(val_n1 < nodes[n0].value || (val_n1 == nodes[n0].value && n1 < n0)) continue;
            
            //Find roots of both nodes.
            r0 = find_root(nodes, n0);
            r1 = find_root(nodes, n1);
            
            //Create the higher node that has not existed.
            if(r1 == -1)
            {                
                //To replace value (-1).
                r1 = n1;
                
                //It is a root of itself.
                nodes[n1].parent = n1;
        
                //Store index of a new component.
                nodes[n1].component = (int)components.size();

                //Create new component.
                Component comp(nodes[n1]);

                //Add index of node in a component.
                comp.inodes.push_back(n1);

                //Add component to the list of components.
                components.push_back(comp);
            }
            
            c0 = nodes[r0].component;
            c1 = nodes[r1].component;

            //If nodes are in the same component.
            if(c0 == c1) continue;
                        
            //Merge two components.
            Component comp_merge(components[c0], components[c1]);
            
            check_when_merging(comp_merge, components[c0], components[c1], val, (int)components.size());

            nodes[r0].parent = r1;
            nodes[r1].component = (int)components.size();
            
            // Tracking voulme/area information of connected components.
            tracking_volume_of_component(time_step, val, comp_merge, volumes, tracking_flag, nodes);
    
            components.push_back(comp_merge);
        }
    }
    
    save_volume_to_file(volumes, time_step);
}

//  Restrict to predefined regions (e.g., lower boundary of the box and land).
template <typename T>
    void generate_regions_in_box(const T *p_land_sea_mask, unsigned long num_rc, unsigned long num_rows, unsigned long num_cols, unsigned int *outputA, unsigned int *outputB)
{
    unsigned int output_region_A_tmp[num_rc];
    
    std::fill_n(output_region_A_tmp, num_rc, 0);

    //Create two sets.
    for(unsigned long i = 0; i < num_rc; ++i)
    {
        //Assigning ones at the same place where is the land-sea mask.
        if(p_land_sea_mask[i] == 1)
            output_region_A_tmp[i] = 1;

        //Assigning ones to the last rows at the bottom of the box.
        if(i < num_cols*0.60)
            outputB[i] = 1;
    }

    //Compute labels for data points using SAUF.
    unsigned int num_comp = sauf(num_rows, num_cols, output_region_A_tmp);

    //Set of size of connected components.
    unsigned int label_counter[num_comp];
    
    std::fill_n(label_counter, num_comp, 0);

    //Calculate size of each component.
    for(unsigned long j = 1; j <= num_comp; j++)
    {
        int counter = 0;

        for(unsigned long i = 0; i < num_rc; ++i)
        {
            if(output_region_A_tmp[i] == j)
            {
                counter++;
            }
        }

        label_counter[j-1] = counter;
    }

    //Take those components that are greater than some arbitrary size value.
    for(unsigned long i = 1; i <= num_comp; ++i)
    {
        //Arbitrary value for skipping components.
        unsigned int skipping_size = 40;

        if(label_counter[i-1] < skipping_size)
        {
            continue;
        }
        else
        {
            for(unsigned long j = 0; j < num_rc; ++j)
            {
                if(output_region_A_tmp[j] == i)
                {
                    outputA[j] = 1;
                }
            }
        }
    }
}

//Build graph on IWV 2d grid.
template <typename T>
    void union_find_alg(const T *input, unsigned long num_rc, unsigned long num_rows,
                    unsigned long num_cols, const_p_teca_variant_array lat,
                    const_p_teca_variant_array lon, unsigned long time_step, const_p_teca_variant_array land_sea_mask)
{

    bool reg_flag = true;

    NESTED_TEMPLATE_DISPATCH_FP(
            const teca_variant_array_impl,
            lat.get(),
            1,

            NESTED_TEMPLATE_DISPATCH(
            const teca_variant_array_impl,
            land_sea_mask.get(),
            2,

            const NT1 *p_lat = dynamic_cast<TT1*>(lat.get())->get();
            const NT1 *p_lon = dynamic_cast<TT1*>(lon.get())->get();

            const NT2 *p_land_sea_mask = dynamic_cast<TT2*>(land_sea_mask.get())->get();

            unsigned int output_region_ls_A_box[num_rc];
            std::fill_n(output_region_ls_A_box, num_rc, 0);
            unsigned int output_region_b_B_box[num_rc];
            std::fill_n(output_region_b_B_box, num_rc, 0);

            if(reg_flag)
            {
                //Generate specific regions of interest.
                generate_regions_in_box(p_land_sea_mask, num_rc, num_rows, num_cols, output_region_ls_A_box, output_region_b_B_box);
            }
            
            //Find connected components based on union-find data structure.
            std::vector<Component> components;
            finding_components_test(components, time_step, output_region_ls_A_box, output_region_b_B_box, num_cols, num_rows, input, p_lon, p_lat);
    ))
}
