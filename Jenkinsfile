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
    stage('Regression Tests') {
      steps {
        sh '''cd build
              ctest -VV'''
      }
    }
    stage('Deploy') {
      steps {
        echo 'Branch: Master.'
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
