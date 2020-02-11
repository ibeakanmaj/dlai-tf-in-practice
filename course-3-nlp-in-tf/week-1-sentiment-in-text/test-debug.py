        return (math.factorial(self.n) / math.factorial(k)*(math.factorial(self.n - k)))
                * (self.p ** k) * (1 - self.p)**(self.n - k)