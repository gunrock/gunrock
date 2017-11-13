pipeline {
  agent any
  stages {
    stage('Init') {
      steps {
        init_git()
      }
    }
    stage('Build') {
      steps {
        cmake_build()
      }
    }
    stage('Unit Tests') {
      steps {
        sh '''cd build
              ./bin/unit_test'''
      }
    }
    stage('Regression Tests') {
      steps {
        sh '''cd build
              cd examples
              ctest -VV'''
      }
    }
    stage('Code Coverage') {
      steps {
        sh '''#!/bin/bash
cd build
CODECOV_TOKEN="d0690e81-c2ed-42d0-8a63-da351c3ae619"
bash <(curl -s https://codecov.io/bash) -t ${CODECOV_TOKEN} || echo "Error: Codecov did not collect coverage reports"'''
      }
    }
    stage('Deploy') {
      steps {
        echo 'Branch: Dev.'
        echo 'Pipleline finished.'
      }
    }
  }
  post {
    always {
      cleanWs()
      
    }
    
  }
}