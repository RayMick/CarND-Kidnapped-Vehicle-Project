/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // A tuning parameter, may affect estimation accuracy, calculation speed
  //cout << "Start particle initialization " << endl;
  // Tried 10, 15, 20, 50, 100 and 200.
  // 15 gives the most stable result and also relative fast calculation speed
  num_particles = 15;

  // This is the initial GPS std deviation
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Normal distribution
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Create a group of normal-distributed particles with mean of initial GPS value and std deviation of GPS signal
  for (int i = 0; i < num_particles; i++) {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
  //cout << "End particle initialization " << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  //cout << "Start particle prediction " << endl;

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // incoporate the motion models
  for (int i = 0; i < num_particles; i++) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;


    double pred_x;
    double pred_y;
    double pred_theta;

    // Checking for very small yaw_rate and consider values below this as "zero" yaw_rate
    if (fabs(yaw_rate) < 1e-6) {
      pred_x = particle_x + velocity * cos(particle_theta) * delta_t;
      pred_y = particle_y + velocity * sin(particle_theta) * delta_t;
      pred_theta = particle_theta;
    }
    else {
      pred_x = particle_x + (velocity / yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
      pred_y = particle_y + (velocity / yaw_rate) * (cos(particle_theta) -  cos(particle_theta + (yaw_rate * delta_t)));
      pred_theta = particle_theta + (yaw_rate * delta_t);
    }

    // Calculate the distribution after applying motion model
    // the post-motion mean is the predicted mean, with the same GPS signal std deviation
    normal_distribution<double> dist_x(pred_x, std_x);
    normal_distribution<double> dist_y(pred_y, std_y);
    normal_distribution<double> dist_theta(pred_theta, std_theta);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);

    //cout << "End particle prediction " << endl;
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // observations -- the landmark coordinates in global map coord transformed/predicted (using homogeneous transformation)
  // from the local car coord w.r.t the particle location
  // predicted -- ground truth (GT) landmark coordinates in global map coord

  ////cout << "Start particle association " << endl;

  // check every predicted/transformed landmark observation to find the nearest one to the current GT landmark
  for (unsigned int i = 0; i < observations.size(); i++) {
    // Initialize a max value as min distance between predicted landmark and observed landmark
    double min_distance = numeric_limits<double>::max();
    // Initialize the ID to a negavite value
    int nearest_pred_landmark_id = -1;

    // iterate through every GT landmark coord to find the smallest distance, i.e., nearest neighbor
    for (unsigned int j = 0; j < predicted.size(); j++) {
      double delta_x_square = (observations[i].x - predicted[j].x) * (observations[i].x - predicted[j].x);
      double delta_y_square = (observations[i].y - predicted[j].y) * (observations[i].y - predicted[j].y);
      double distance = sqrt(delta_x_square + delta_y_square);

      // Update the min distance to keep track of the nearest neighbor
      if (distance < min_distance) {
        min_distance = distance;
        nearest_pred_landmark_id = predicted[j].id;
      }
    }

    // each ground truth landmark may pair to a single transformed landmark, even though there may be several transformed landmark close to the same ground truth landmark
    observations[i].id = nearest_pred_landmark_id;

    ////cout << "End particle association " << endl;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  //cout << "Start particle weight update " << endl;

  double sum_weights = 0.0;
  // Do the following process for all the particles
  for (int i = 0; i < num_particles; i++) {
    //cout << "particle id is " << i << endl;
    double particles_x = particles[i].x;
    double particles_y = particles[i].y;
    double theta = particles[i].theta;

    double sensor_range_sqrt = sqrt(sensor_range * sensor_range);
    vector<LandmarkObs> filtered_landmarks;
    // ------------------------------------------------------------------------------------------
    // Step 1: Select the landmarks within the sensor range
    // ------------------------------------------------------------------------------------------
    //cout << "Start step 1 of particle weight update " << endl;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      int id = map_landmarks.landmark_list[j].id_i;
      double delta_x = particles_x - landmark_x;
      double delta_y = particles_y - landmark_y;
      ////cout << "landmark_x, landmark_y is " << landmark_y << ", " << landmark_y << endl;
      if (sqrt(delta_x * delta_x + delta_y * delta_y) <= sensor_range_sqrt) {
        //cout << "landmark_x, landmark_y is " << landmark_x << ", " << landmark_y << endl;
        filtered_landmarks.push_back(LandmarkObs{id, landmark_x, landmark_y});
      }
    }

    // ------------------------------------------------------------------------------------------
    // Step 2: Transform the landmark measurement from car coordinate system to global map system
    // ------------------------------------------------------------------------------------------
    //cout << "Start step 2 of particle weight update " << endl;
    vector<LandmarkObs> transformed_obs_list;
    for (unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs transformed_obs;
      transformed_obs.x = particles_x + cos(theta) * observations[j].x - sin(theta) * observations[j].y;
      transformed_obs.y = particles_y + sin(theta) * observations[j].x + cos(theta) * observations[j].y;
      //transformed_obs.id = observations[j].id;
      transformed_obs.id = j;
      transformed_obs_list.push_back(transformed_obs);
    }

    // ------------------------------------------------------------------------------------------
    // Step 3: Associate the
    // ------------------------------------------------------------------------------------------
    //cout << "Start step 3 of particle weight update " << endl;
    dataAssociation(filtered_landmarks, transformed_obs_list);

    // ------------------------------------------------------------------------------------------
    // Step 4: Weight update
    // ------------------------------------------------------------------------------------------
    //cout << "Start step 4 of particle weight update " << endl;
    particles[i].weight = 1.0;

    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_square = sigma_x * sigma_x;
    double sigma_y_square = sigma_y * sigma_y;
    double gauss_norm = (1.0/(2.0 * M_PI * sigma_x * sigma_y));

    for (unsigned int j = 0; j < transformed_obs_list.size(); j++) {
      double obs_x = transformed_obs_list[j].x;
      double obs_y = transformed_obs_list[j].y;
      double obs_id = transformed_obs_list[j].id;
      double total_prob = 1.0;

      for (int k = 0; k < filtered_landmarks[k].x; k++) {
        double filtered_landmark_x = filtered_landmarks[k].x;
        double filtered_landmark_y = filtered_landmarks[k].y;
        double filtered_landmark_id = filtered_landmarks[k].id;

        if (obs_id == filtered_landmark_id) {
          //cout << "inner obs_id is " << obs_id << endl;
          //cout << "inner filtered_landmark_id is " << filtered_landmark_id << endl;

          //cout << "obs_x, obs_y is " << obs_x << ", " << obs_y << endl;
          //cout << "filtered_landmark_x, filtered_landmark_y is " << filtered_landmark_x << ", " << filtered_landmark_y << endl;

          double exponent = ((pow((obs_x - filtered_landmark_x), 2))/(2.0 * sigma_x_square)) + ((pow((obs_y - filtered_landmark_y), 2))/(2.0 * sigma_y_square));
          //total_prob = gauss_norm * exp(-1.0 * ((pow((obs_x - filtered_landmark_x), 2))/(2.0 * sigma_x_square)) + ((pow((obs_y - filtered_landmark_y), 2))/(2.0 * sigma_y_square)));
          total_prob = gauss_norm * exp(-1.0 * exponent);
          //cout << "inner exponent is " << exponent << endl;
          //cout << "inner gauss_norm is " << gauss_norm << endl;
          //cout << "inner exp(-1.0 * exponent) is " << exp(-exponent) << endl;
          //cout << "inner total_prob is " << total_prob << endl;
          particles[i].weight *= total_prob;
        }
      }
    }
    sum_weights += particles[i].weight;
    //cout << "inner sum_weights summation is " << sum_weights << endl;
  }

  //cout << "sum_weights is " << sum_weights << endl;
  // Normalize the weight, sum_weights is the total of the probability of all particles
  for (unsigned int i = 0; i < particles.size(); i++) {
    //cout << "particles[i].weight is " << particles[i].weight << endl;
    //cout << "   11111--sum_weights is " << sum_weights << endl;
    particles[i].weight /= sum_weights;
    weights[i] = particles[i].weight;
  }

  //cout << "End particle weight update " << endl;

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  //cout << "Start particle resampling " << endl;

  vector<Particle> resampled_particles;
  vector<double> weights;

  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // random index for the resampling wheel
  uniform_int_distribution<int> ind_dist(0, num_particles - 1);
  auto index = ind_dist(gen);

  // get the max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> real_dist(0.0, max_weight);

  // resampling wheel algorithm
  double beta = 0.0;
  for (int i = 0; i < num_particles; i++) {
    beta += real_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;

  //cout << "End particle resampling " << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
