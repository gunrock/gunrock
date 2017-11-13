// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// initialize source codes
def init_git() {
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init'
    }
  }
}

// build gunrock using cmake
def cmake_build() {
  checkout scm
  retry(5) {
    timeout(time: 20, unit: 'MINUTES') {
      sh 'mkdir -p build'
      sh '''cd build
            cmake ..
            make -j16'''
    }
  }
}

// code coverage access codecov.io
def codecov() {
  timeout(time: 20, unit: 'MINUTES') {
      sh '''cd build
            CODECOV_TOKEN="d0690e81-c2ed-42d0-8a63-da351c3ae619"
            bash <(curl -s https://codecov.io/bash) -t ${CODECOV_TOKEN} || echo "Error: Codecov did not collect coverage reports"'''
    }
}

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
        codecov()
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
